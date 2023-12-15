import numpy as np
import polars as pl
import pandas as pd

def match_detections(gt_steps, pred_scores, pred_steps, tolerance):

    sort_idxs = np.argsort(pred_scores)[::-1]
    pred_steps = pred_steps[sort_idxs]
    pred_scores = pred_scores[sort_idxs]
    
    is_matched = np.full_like(pred_steps, False, dtype=bool)
    gts_matched = set()
    gt_steps = set(gt_steps)
    
    for i, pred_step in enumerate(pred_steps):
        best_error = tolerance
        best_gt = None

        #### To implement removing iterating from the inner for loop by deleting keys
        gt_steps_iters = gt_steps.copy()
        
        for gt_step in gt_steps_iters:
            error = abs(gt_step-pred_step)
            if error < best_error:
                best_gt = gt_step
                best_error = error
                gt_steps.remove(gt_step)

        if best_gt is not None:
            is_matched[i] = True
            
    return pred_scores, is_matched

def precision_recall_curve(matches: np.ndarray, scores: np.ndarray, p: int):
    if len(matches) == 0:
        return [1], [0], []

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind='stable')[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / p  # total number of ground truths might be different than total number of matches

    # Stop when full recall attained and reverse the outputs so recall is non-increasing.
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def comp_scorer(true_df, pred_df, tolerances):
    
    true_df_parts = true_df.partition_by(by=['series_id', 'event'], maintain_order=True, as_dict=True)
    pred_df_parts = pred_df.partition_by(by=['series_id', 'event'], maintain_order=True, as_dict=True)
    unq_ser_ids = true_df['series_id'].unique()

    class_counts_dct = {k: v for k, v in true_df['event'].value_counts().to_numpy()}

    scores_events = []
    for ev_name in ['onset', 'wakeup']:
        score_tolerances_lst = []
        for tol in tolerances[ev_name]:
            pred_scores_lst = [] 
            pred_matches_lst = []


            for ser_id in unq_ser_ids:
                true_df_part = true_df_parts[(ser_id, ev_name)]
                
                if (ser_id, ev_name) in pred_df_parts:
                    pred_df_part = pred_df_parts[(ser_id, ev_name)]
                else:
                    pred_df_part = pl.DataFrame(pd.DataFrame(columns=pred_df.columns))  
                pred_scores, pred_matches =  match_detections(true_df_part['step'].to_numpy(), pred_df_part['score'].to_numpy(), pred_df_part['step'].to_numpy(), tol)
                pred_scores_lst.append(pred_scores)
                pred_matches_lst.append(pred_matches)

            pred_scores_all = np.concatenate(pred_scores_lst)
            pred_matches_all = np.concatenate(pred_matches_lst)


            score_tolerance = average_precision_score(pred_matches_all, pred_scores_all, class_counts_dct[ev_name])
            score_tolerances_lst.append(score_tolerance)

        mean_tolerance_score = np.mean(score_tolerances_lst)

        scores_events.append(mean_tolerance_score)

    return np.mean(scores_events)