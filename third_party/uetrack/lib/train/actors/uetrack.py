from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap
import torch.nn.functional as F
from lib.utils.gs import gumbel_softmax

class UETrack_Actor(BaseActor):
    """ Actor for training the sutrack"""
    def __init__(self, net, objective, loss_weight, settings, cfg, net_teacher,adaptive_net=None):
        super().__init__(net, objective)
        self.net_teacher = net_teacher
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        self.distill_logits_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.adaptive_net = adaptive_net
        self.num_distill_layer = len(cfg.TRAIN.DISTILL_LAYER_T)

        self.dist_dict = {'TEMPERATURE':cfg.TRAIN.TEMPERATURE}
        print ('Distill Setting: ', f'TEMPERATURE: {self.dist_dict["TEMPERATURE"]}' )


    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict_teacher,out_dict_student,feature_list_teacher,feature_list_student,feature_list_student_aligned,active_layer,output = self.forward_pass(data)
        # compute losses
        loss, loss_adapt, status = self.compute_losses(out_dict_teacher, out_dict_student,feature_list_teacher,feature_list_student_aligned, active_layer, output, data)
        return loss, loss_adapt, status

    def forward_pass(self, data):
        b = data['search_images'].shape[1]   # n,b,c,h,w
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)  # (n*b, c, h, w)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)

        if self.multi_modal_language:
            text = data['nlp_ids'].permute (1,0)
            text_src_teacher,text_src = self.net_teacher(text_data=text, mode='text')
            text_src_student = self.net(text_src=text_src,mode='text')
        else:
            text_src_teacher = None
            text_src_student = None

        # task_class
        task_index_batch = [self.cfg.MODEL.TASK_INDEX[key.upper()] for key in data['dataset']]
        task_index_batch = torch.tensor(task_index_batch).cuda() #torch.Size([bs])
        #forward teacher
        enc_opt_teacher,feature_list_teacher = self.net_teacher(template_list=template_list,
                           search_list=search_list,
                           template_anno_list=template_anno_list,
                           text_src=text_src_teacher,
                           task_index=task_index_batch,
                           mode='encoder') # forward the encoder#[#cls,x,z,text]
        outputs_teacher, task_class_output_teacher = self.net_teacher(feature=enc_opt_teacher, mode="decoder")
        task_class_output_teacher = task_class_output_teacher.view(-1, task_class_output_teacher.size(-1))
        outputs_teacher['task_class'] = task_class_output_teacher
        outputs_teacher['task_class_label'] = task_index_batch
        #forward student
        enc_opt_student,feature_list_student,feature_list_student_aligned = self.net(template_list=template_list,
                           search_list=search_list,
                           template_anno_list=template_anno_list,
                           text_src=text_src_student,
                           task_index=task_index_batch,
                           mode='encoder') # forward the encoder#[#cls,x,z,text]
        outputs_student, task_class_output_student = self.net(feature=enc_opt_student, mode="decoder")
        task_class_output_student = task_class_output_student.view(-1, task_class_output_student.size(-1))
        outputs_student['task_class'] = task_class_output_student
        outputs_student['task_class_label'] = task_index_batch

        # forward adaptive net
        feature_list_teacher_detach = [feature_list_teacher[-1].detach()]
        feature_list_student_detach = [feature_list_student[-1].detach()]
        adaptive_layer = self.adaptive_net(feat_t_list=feature_list_teacher_detach,
                                           feat_s_list=feature_list_student_detach)  # [[B,2],[B,2]......]
        adaptive_layer = torch.stack(adaptive_layer).permute(1, 0, 2)
        active_layer = gumbel_softmax(adaptive_layer, temperature=5)  # [b,N_LAYER,2],one_hot
        out_ac = [active_layer[:, i, 0].contiguous().float().view(-1) for i in
                     range(active_layer.shape[1])]  # 取出第0维度
        out_ac = out_ac[-1]
        outputs = {}
        for key in outputs_teacher.keys():
            outputs_student_adapt_value = outputs_student[key].detach()
            outputs_teacher_adapt_value = outputs_teacher[key].detach()
            out_ac_re = out_ac.view([out_ac.size(0)] + [1] * (outputs_student_adapt_value.dim() - 1))
            if key != "task_class_label":
                outputs[key] = (1 - out_ac_re) * outputs_student_adapt_value + out_ac_re * \
                               outputs_teacher_adapt_value
            else:
                outputs[key] = task_index_batch
        return outputs_teacher,outputs_student,feature_list_teacher,feature_list_student,feature_list_student_aligned,active_layer,outputs

    def compute_losses(self, pred_dict_teacher, pred_dict_student, feature_list_teacher,feature_list_student,active_layer,output,
                       gt_dict, return_status=True):
        # task classification loss
        task_cls_loss = self.objective['task_cls'](pred_dict_student['task_class'], pred_dict_student['task_class_label'])

        task_cls_loss_adapt = self.objective['task_cls'](output['task_class'],output['task_class_label'])

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE) # list of torch.Size([b, H, W])
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1) # torch.Size([b, 1, H, W])

        # Get boxes
        pred_boxes = pred_dict_student['pred_boxes'] # torch.Size([b, 1, 4])
        pred_boxes_teacher = pred_dict_teacher['pred_boxes']
        pred_boxes_adapt = output['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_teacher_vec = box_cxcywh_to_xyxy(pred_boxes_teacher).view(-1, 4)
        pred_boxes_adapt_vec = box_cxcywh_to_xyxy(pred_boxes_adapt).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            giou_loss_adapt, iou_adapt = self.objective['giou'](pred_boxes_adapt_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            giou_loss_adapt, iou_adapt = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        _, iou_teacher = self.objective['giou'](pred_boxes_teacher_vec, gt_boxes_vec)
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        l1_loss_adapt = self.objective['l1'](pred_boxes_adapt_vec, gt_boxes_vec)
        # compute location loss
        if 'score_map' in pred_dict_student:
            location_loss = self.objective['focal'](pred_dict_student['score_map'], gt_gaussian_maps)
            location_loss_adapt = self.objective['focal'](output['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
            location_loss_adapt = torch.tensor(0.0, device=l1_loss.device)

        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss +
                self.loss_weight['task_cls'] * task_cls_loss)
        loss_adapt = (self.loss_weight['giou'] * giou_loss_adapt +
                self.loss_weight['l1'] * l1_loss_adapt +
                self.loss_weight['focal'] * location_loss_adapt +
                self.loss_weight['task_cls'] * task_cls_loss_adapt)

        if loss_adapt > loss:
            loss_adapt *=2
        if loss_adapt < loss:
            loss_adapt /=2
        ac_middle = [active_layer[:, i, 0].contiguous().float().view(-1) for i in range(active_layer.shape[1])]
        loss_adapt += max(0, pred_boxes_adapt.shape[0] / 2 - sum(ac_middle[-1]))

        kd_loss,feat_loss = self.compute_losses_distill(pred_dict_teacher, pred_dict_student, feature_list_teacher,feature_list_student,active_layer)
        # weighted sum
        loss = (loss +
                self.loss_weight['kd'] * kd_loss +
                self.loss_weight['feat'] * feat_loss)

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            mean_iou_teacher = iou_teacher.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/task_class": task_cls_loss.item(),
                      "Loss/kd": kd_loss.item(),
                      "Loss/feat": feat_loss.item(),
                      "loss_adaptive": loss_adapt.item(),
                      "IoU_student": mean_iou.item(),
                      "IoU_teacher": mean_iou_teacher.item(),
                      }
            return loss, loss_adapt, status
        else:
            return loss, loss_adapt

    def compute_losses_distill(self, pred_dict_teacher, pred_dict_student, feature_list_teacher,feature_list_student,active_layer):
        kd_loss = torch.tensor(0.0, device=pred_dict_teacher['score_map'].device)
        feat_loss = torch.tensor(0.0, device=pred_dict_teacher['score_map'].device)

        ac_middle = [active_layer[:, i, 0].contiguous().float().view(-1) for i in range(active_layer.shape[1])]  # 取出第0维度

        # distill logit loss
        T = self.dist_dict['TEMPERATURE']
        teacher_logits = pred_dict_teacher['score_map'].flatten(start_dim=1)
        student_logits = pred_dict_student['score_map'].flatten(start_dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        student_probs = F.log_softmax(student_logits / T, dim=1)
        last_choose = ac_middle[-1].detach()
        choose_index = last_choose > 0.5
        if student_logits[choose_index].shape[0] > 0:  # kd_loss
            kd_loss += (self.distill_logits_loss(student_probs[choose_index],teacher_probs[choose_index])*(T*T)
                        * student_logits[choose_index].shape[0] / student_logits.shape[0])

        # distill feature loss
        # cls,x,z,text
        if (len(feature_list_student) != len(ac_middle) and len(ac_middle) == 1):
            ac_middle = ac_middle*len(feature_list_student)
        middle_choose_list = [each_ac.detach() for each_ac in ac_middle]
        feature_list_student_ = [feature_list_student[feat][middle_choose_list[feat] > 0.5] if
               feature_list_student[feat][middle_choose_list[feat] > 0.5].shape[0] > 0 else torch.zeros_like(feature_list_student[feat]).cuda()
               for feat in range(len(feature_list_student))]
        feature_list_teacher_ = [feature_list_teacher[feat][middle_choose_list[feat] > 0.5] if
               feature_list_teacher[feat][middle_choose_list[feat] > 0.5].shape[0] > 0 else torch.zeros_like(feature_list_teacher[feat]).cuda()
               for feat in range(len(feature_list_teacher))]
        feat_loss_list = []
        for ft_i,fs_i in zip(feature_list_teacher_,feature_list_student_):
            feat_loss_i = F.mse_loss(ft_i,fs_i)
            feat_loss_list.append(feat_loss_i)
        feat_loss_list = [feat_loss_list[each_kd] * feature_list_student[0][middle_choose_list[each_kd] > 0.5].size(0) / feature_list_student[0].size(0) if
                 feature_list_student_[each_kd].shape[0] > 0
                 else torch.tensor(0.0, dtype=torch.float).cuda()
                 for each_kd in range(len(feat_loss_list))]
        feat_loss += sum(feat_loss_list)/len(feat_loss_list)
        return kd_loss,feat_loss
