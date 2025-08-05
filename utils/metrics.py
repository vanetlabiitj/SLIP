import csv

import numpy as np
from sklearn.metrics import recall_score, f1_score


def region_wise_metric(labels, predictions, outputs):
  macro_d, micro_d, macro_m, micro_m, macro_s, micro_s = [], [], [], [], [], []
  tpr_d, tpr_m, tpr_s = [], [], []
  sf1_d, sf1_m, sf1_s = [], [], []

  # dense_len = 23
  # mid_len = 33
  # tail_len = 57
  labels = np.reshape(labels, (-1, 113, 8))
  predictions = np.reshape(predictions, (-1, 113, 8))
  outputs = np.reshape(outputs, (-1, 113, 8))

  severity_weights = [2.0, 3.0, 4.0, 5.0, 7.0, 1.0, 6.0, 8.0]  # Higher = more important
  f1_per_class_all_sectors = []  # List to store per-sector F1-per-class arrays
  sector_ids = []

  with open("RFL_f1_per_class_per_sector.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header_written = False

    for sector_idx in range(labels.shape[1]):
      y_true_sector = labels[:, sector_idx, :]  # True labels for current sector
      y_pred_sector = predictions[:, sector_idx, :]  # Predicted labels for current sector

      # Calculate micro F1 score for current sector
      micro_f1_sector = f1_score(y_true_sector, y_pred_sector, average='micro')
      macro_f1_sector = f1_score(y_true_sector, y_pred_sector, average='macro')
      recall_per_class = recall_score(y_true_sector, y_pred_sector, average='macro')
      f1_per_class = f1_score(y_true_sector, y_pred_sector, average=None)  # Shape: (num_classes,)
      f1_per_class_all_sectors.append(f1_per_class)
      sector_ids.append(sector_idx)

      # Normalize severity weights to sum to 1
      normalized_weights = [w / sum(severity_weights) for w in severity_weights]

      # Compute custom weighted F1
      severity_weighted_f1 = sum(f * w for f, w in zip(f1_per_class, normalized_weights))

      # Write CSV header once
      if not header_written:
            class_headers = [f"class_{i}" for i in range(len(f1_per_class))]
            writer.writerow(["sector_idx"] + class_headers)
            header_written = True

      writer.writerow([sector_idx] + list(f1_per_class))

      if sector_idx in [3, 15, 0, 10, 26, 39, 25, 8, 58, 23, 6, 33, 43, 49, 11, 50, 20, 7, 30, 32, 2, 75, 56]:
            macro_d.append(macro_f1_sector)
            micro_d.append(micro_f1_sector)
            tpr_d.append(recall_per_class)
            sf1_d.append(severity_weighted_f1)

      elif sector_idx in [37, 52, 65, 36, 46, 22, 18, 24, 48, 19, 9, 64, 53, 71, 55, 29, 1, 4, 87, 93, 60, 63, 40, 81, 79, 72, 38, 91, 28, 83, 31, 68, 16]:
            macro_m.append(macro_f1_sector)
            micro_m.append(micro_f1_sector)
            tpr_m.append(recall_per_class)
            sf1_m.append(severity_weighted_f1)

      else:
            macro_s.append(macro_f1_sector)
            micro_s.append(micro_f1_sector)
            tpr_s.append(recall_per_class)
            sf1_s.append(severity_weighted_f1)


  avg_macro_d = sum(macro_d)/len(macro_d)
  avg_micro_d = sum(micro_d)/len(micro_d)
  avg_macro_m = sum(macro_m)/len(macro_m)
  avg_micro_m = sum(micro_m)/len(micro_m)
  avg_macro_s = sum(macro_s)/len(macro_s)
  avg_micro_s = sum(micro_s)/len(micro_s)

  # average tpr
  avg_tpr_d = sum(tpr_d)/len(tpr_d)
  avg_tpr_m = sum(tpr_m)/len(tpr_m)
  avg_tpr_s = sum(tpr_s)/len(tpr_s)

  avg_sf1_d = sum(sf1_d)/len(sf1_d)
  avg_sf1_m = sum(sf1_m)/len(sf1_m)
  avg_sf1_s = sum(sf1_s)/len(sf1_s)


  print(".........................Region Metrics........................\n")
  print(f"Macro_dense = {avg_macro_d}, Micro_dense = {avg_micro_d}, Macro_mid = {avg_macro_m}, Micro_mid = {avg_micro_m}, Macro_tail = {avg_macro_s}, Micro_tail = {avg_micro_s} ")
  print(f"SF1_dense = {avg_sf1_d}, SF1_mid = {avg_sf1_m}, SF1_tail = {avg_sf1_s}")
  print(f"TPR_dense = {avg_tpr_d}, TPR_mid = {avg_tpr_m}, TPR_tail = {avg_tpr_s}")
  print(".........................Fairness Metrics........................\n")
  print(f"Regional Disparity = {avg_macro_d - avg_macro_s/ (avg_macro_d + avg_macro_s)}")
  print(f"Equal Opportunity = {(avg_tpr_d + avg_tpr_s + avg_tpr_d )/3}")

  # Compute and display average F1-per-class across all sectors
  f1_per_class_all_sectors = np.array(f1_per_class_all_sectors)
  avg_f1_per_class = np.mean(f1_per_class_all_sectors, axis=0)

  print("Average F1-score per class across all sectors:")
  for i, f1_val in enumerate(avg_f1_per_class):
     print(f"Class {i}: {f1_val:.4f}")

