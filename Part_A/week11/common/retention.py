
def calculate_gt_retention(top_k_lines, gt_lines):
    selected_lines = set(top_k_lines)

    retained_gt_lines = gt_lines & selected_lines

    if len(gt_lines) == 0:
        retention = None
    else:
        retention = len(retained_gt_lines) / len(gt_lines)

    return retained_gt_lines, retention
