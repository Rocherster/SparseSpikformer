"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F

class DistillDiffPruningLoss_dynamic(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5, dynamic=False, pruning_loc=[4,5,6], keep_ratio=[0.75, 0.5, 0.25], clf_weight=0, mse_token=False, print_mode=True):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic

        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight

        print('ratio_weight=', ratio_weight, 'distill_weight', distill_weight)


        if dynamic:
            print('using dynamic loss')

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred, mask, out_pred_score = outputs

        pred_loss = 0.0

        ratio = self.keep_ratio
        for i, score in enumerate(out_pred_score):
            if self.dynamic:
                pos_ratio = score.mean()
            else:
                pos_ratio = score.mean(1)
            pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()

        cls_loss = self.base_criterion(pred, labels)

        with torch.no_grad():
            cls_t = self.teacher_model(inputs)

        cls_kl_loss = F.kl_div(
                F.log_softmax(pred, dim=-1),
                F.log_softmax(cls_t, dim=-1),
                reduction='batchmean',
                log_target=True
            )

        loss_part = []
        
        # print(cls_loss, pred_loss)
        loss = self.clf_weight * cls_loss + self.ratio_weight * pred_loss / len(self.pruning_loc) + self.distill_weight * cls_kl_loss 

        # if self.print_mode:  # True
        #     self.cls_loss += cls_loss.item()
        #     self.ratio_loss += pred_loss.item()
        #     self.cls_distill_loss += cls_kl_loss.item()
        #     loss_part.append(cls_loss)
        #     loss_part.append(pred_loss)
        #     loss_part.append(cls_kl_loss)
        #     self.count += 1
        #     if self.count == 100:
        #         print('loss info: cls_loss=%.4f, ratio_loss=%.4f, cls_kl=%.4f' % (self.cls_loss / 100, self.ratio_loss / 100, self.cls_distill_loss/ 100))
        #         self.count = 0
        #         self.cls_loss = 0
        #         self.ratio_loss = 0
        #         self.cls_distill_loss = 0
        return loss, loss_part

