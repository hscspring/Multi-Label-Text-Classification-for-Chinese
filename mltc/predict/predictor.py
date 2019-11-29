import torch
import numpy as np
from utils.utils import model_device
from callback.progressbar import ProgressBar
from sklearn.metrics import f1_score


class Predictor(object):
    def __init__(self,
                 model,
                 logger,
                 n_gpu
                 ):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu=n_gpu, model=self.model)

    def predict(self, data, thresh):
        pbar = ProgressBar(n_total=len(data))
        all_logits = None
        # y_true = torch.LongTensor()
        y_true = None
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # y_true = torch.cat((y_true, label_ids), 0)
                if y_true is None:
                    y_true = label_ids.detach().cpu().numpy()
                else:
                    y_true = np.concatenate(
                        [y_true, label_ids.detach().cpu().numpy()], axis=0)
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()
                if all_logits is None:
                    all_logits = logits.detach().cpu().numpy()
                else:
                    all_logits = np.concatenate(
                        [all_logits, logits.detach().cpu().numpy()], axis=0)
                pbar.batch_step(step=step, info={}, bar_type='Testing')
        y_pred = (all_logits > thresh) * 1
        micro = f1_score(y_true, y_pred, average='micro')
        macro = f1_score(y_true, y_pred, average='macro')
        score = (micro + macro) / 2
        self.logger.info(
            "\nScore: micro {}, macro {} Average {}".format(
                micro, macro, score))
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return all_logits, y_pred
