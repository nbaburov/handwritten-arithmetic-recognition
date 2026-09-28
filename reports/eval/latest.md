# Eval Summary

Run: 20260622T142513
Config: real_iter11

## Aggregate Metrics

| Metric | Current |
|--------|---------|
| n_samples | 190 |
| row_acc_macro [PRIMARY] | 0.8542 |
| col_acc_macro [PRIMARY] | 0.7915 |
| n_spatial_samples | 189 |
| eq_kind_acc | 0.5474 |
| ood_rate | 0.0158 |
| ood_honest_rate | 0.0158 |
| mean_label_acc | 0.8900 |
| mean_iou_matched | n/a |
| mean_recall | 0.9817 |
| total_wall_ms (ms) | 9786.9 |

## Per Fine-Label (recall + top confusion)

| fine_label | n | recall | top confusions (pred:count) |
|---|---|---|---|
| carry_4 | 3 | 0.0000 | main_4:2, main_9:1 |
| borrow_4 | 3 | 0.3333 | main_4:1, borrow_9:1 |
| borrow_0 | 3 | 0.3333 | main_0:2 |
| carry_3 | 2 | 0.5000 | carry_1:1 |
| borrow_8 | 4 | 0.5000 | borrow_2:1, borrow_3:1 |
| op_divide | 8 | 0.6250 | main_5:1, main_2:1, op_plus:1 |
| carry_2 | 8 | 0.6250 | main_2:3 |
| main_1 | 134 | 0.6716 | main_7:26, main_9:13, borrow_1:2 |
| main_4 | 152 | 0.7039 | main_9:42, main_1:1, main_8:1 |
| borrow_1 | 25 | 0.8000 | carry_1:3, borrow_9:2 |
| carry_1 | 46 | 0.8261 | main_1:5, carry_9:1, carry_7:1 |
| main_8 | 83 | 0.8313 | main_0:9, main_5:2, main_3:1 |
| main_7 | 109 | 0.8624 | main_2:9, main_1:3, div_bracket:1 |
| div_bracket | 36 | 0.9167 | main_8:1, main_4:1, main_5:1 |
| op_plus | 60 | 0.9333 | main_4:1, op_minus:1, main_1:1 |
| op_times | 33 | 0.9394 | main_1:2 |
| result_bar | 114 | 0.9474 | op_divide:3, main_7:1, op_minus:1 |
| main_3 | 133 | 0.9549 | main_9:2, borrow_3:1, main_2:1 |
| main_0 | 90 | 0.9556 | borrow_0:2, main_2:1, main_6:1 |
| op_minus | 31 | 0.9677 | main_3:1 |
| main_2 | 115 | 0.9739 | main_1:3 |
| main_9 | 99 | 0.9899 | main_0:1 |
| main_5 | 104 | 0.9904 | main_2:1 |
| main_6 | 110 | 0.9909 | borrow_6:1 |
| carry_0 | 1 | 1.0000 | - |
| carry_6 | 1 | 1.0000 | - |
| carry_5 | 1 | 1.0000 | - |
| borrow_3 | 2 | 1.0000 | - |
| borrow_2 | 5 | 1.0000 | - |
| borrow_6 | 1 | 1.0000 | - |
| borrow_9 | 1 | 1.0000 | - |

## Per Equation-Kind

| equation_kind | n | eq_kind_acc | CI_lo | CI_hi | row_acc_macro | CI_lo | CI_hi | col_acc_macro | CI_lo | CI_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| addition | 36 | 0.3611 | 0.2248 | 0.5243 | 0.9691 | 0.8583 | 0.9951 | 0.7966 | 0.6497 | 0.9025 |
| bare_digits | 46 | 0.9565 | 0.8547 | 0.9880 | 0.9000 | 0.7696 | 0.9527 | 0.9167 | 0.7968 | 0.9657 |
| division | 36 | 0.5556 | 0.3958 | 0.7046 | 0.6717 | 0.5033 | 0.7979 | 0.6051 | 0.4486 | 0.7522 |
| multiplication | 36 | 0.6389 | 0.4757 | 0.7752 | 0.7778 | 0.6191 | 0.8828 | 0.7884 | 0.6191 | 0.8828 |
| subtraction | 36 | 0.1111 | 0.0441 | 0.2532 | 0.9406 | 0.8186 | 0.9846 | 0.8193 | 0.6497 | 0.9025 |

## Per Scene-Case (diagnostic, no CI claim)

| scene_case | n | eq_kind_acc | row_acc_macro | col_acc_macro |
|---|---|---|---|---|
| addition | 17 | 0.4118 | 0.9869 | 0.8763 |
| addition_dense_carries | 6 | 0.3333 | 1.0000 | 0.6778 |
| addition_no_bar | 3 | 0.0000 | 0.8889 | 0.7619 |
| addition_no_carries | 5 | 0.6000 | 0.9556 | 0.7956 |
| addition_op_right | 5 | 0.2000 | 0.9333 | 0.6905 |
| bare_digit_grid | 10 | 1.0000 | 1.0000 | 0.7250 |
| bare_digits | 12 | 1.0000 | 0.6250 | 1.0000 |
| division-long | 11 | 0.7273 | 0.2903 | 0.2608 |
| division-short | 17 | 0.5294 | 0.7640 | 0.6775 |
| division-simple | 8 | 0.3750 | 1.0000 | 0.9250 |
| multiplication-multi | 14 | 0.7857 | 0.4883 | 0.6452 |
| multiplication-simple | 14 | 0.5714 | 0.9762 | 0.8107 |
| multiplication_simple_no_bar | 4 | 0.2500 | 0.8750 | 1.0000 |
| multiplication_simple_op_right | 4 | 0.7500 | 1.0000 | 1.0000 |
| standalone_bar | 12 | 0.9167 | 1.0000 | 0.9167 |
| standalone_bracket | 12 | 0.9167 | 1.0000 | 1.0000 |
| subtraction | 16 | 0.1250 | 0.9393 | 0.7308 |
| subtraction_heavy_borrow | 10 | 0.2000 | 1.0000 | 0.9833 |
| subtraction_no_bar | 4 | 0.0000 | 0.7917 | 1.0000 |
| subtraction_op_right | 6 | 0.0000 | 0.9444 | 0.6616 |

## Per-Sample Results

| Sample | GT | Pred | OK | row_acc | col_acc | OOD Reason | Wall (ms) |
|--------|----|------|----|---------|---------|------------|-----------|
| re-addition-000 | addition | add | yes | 1.0000 | 1.0000 |  | 83.2 |
| re-addition-001 | addition | multiply | no | 1.0000 | 1.0000 |  | 44.1 |
| re-addition-002 | addition | subtract | no | 1.0000 | 0.5714 |  | 42.8 |
| re-addition-003 | addition | add | yes | 1.0000 | 1.0000 |  | 42.8 |
| re-addition-004 | addition | multiply | no | 1.0000 | 1.0000 |  | 44.2 |
| re-addition-005 | addition | add | yes | 1.0000 | 1.0000 |  | 48.1 |
| re-addition-006 | addition | add | yes | 1.0000 | 1.0000 |  | 62.5 |
| re-addition-007 | addition | subtract | no | 1.0000 | 0.6250 |  | 83.1 |
| re-addition-008 | addition | divide | no | 1.0000 | 1.0000 |  | 41.6 |
| re-addition-009 | addition | multiply | no | 1.0000 | 0.5000 |  | 42.4 |
| re-addition-010 | addition | add | yes | 1.0000 | 1.0000 |  | 51.3 |
| re-addition-011 | addition | subtract | no | 1.0000 | 1.0000 |  | 44.7 |
| re-addition-012 | addition | subtract | no | 1.0000 | 0.8000 |  | 44.6 |
| re-addition-013 | addition | add | yes | 1.0000 | 1.0000 |  | 42.3 |
| re-addition-014 | addition | add | yes | 1.0000 | 1.0000 |  | 81.9 |
| re-addition-015 | addition | subtract | no | 1.0000 | 0.4000 |  | 43.6 |
| re-addition-016 | addition | multiply | no | 0.7778 | 1.0000 |  | 44.2 |
| re-addition_dense_carries-000 | addition | add | yes | 1.0000 | 0.4667 |  | 49.5 |
| re-addition_dense_carries-001 | addition | subtract | no | 1.0000 | 0.8333 |  | 46.7 |
| re-addition_dense_carries-002 | addition | subtract | no | 1.0000 | 0.6667 |  | 43.6 |
| re-addition_dense_carries-003 | addition | subtract | no | 1.0000 | 0.1000 |  | 42.8 |
| re-addition_dense_carries-004 | addition | multiply | no | 1.0000 | 1.0000 |  | 43.1 |
| re-addition_dense_carries-005 | addition | add | yes | 1.0000 | 1.0000 |  | 46.7 |
| re-addition_no_bar-000 | addition | subtract | no | 1.0000 | 0.2857 |  | 43.2 |
| re-addition_no_bar-001 | addition | divide | no | 1.0000 | 1.0000 |  | 43.9 |
| re-addition_no_bar-002 | addition | bare_digits | no | 0.6667 | 1.0000 |  | 41.0 |
| re-addition_no_carries-000 | addition | subtract | no | 1.0000 | 0.2000 |  | 43.5 |
| re-addition_no_carries-001 | addition | add | yes | 0.7778 | 0.7778 |  | 81.8 |
| re-addition_no_carries-002 | addition | subtract | no | 1.0000 | 1.0000 |  | 43.7 |
| re-addition_no_carries-003 | addition | add | yes | 1.0000 | 1.0000 |  | 42.3 |
| re-addition_no_carries-004 | addition | add | yes | 1.0000 | 1.0000 |  | 45.2 |
| re-addition_op_right-000 | addition | multiply | no | 1.0000 | 1.0000 |  | 43.4 |
| re-addition_op_right-001 | addition | subtract | no | 1.0000 | 1.0000 |  | 45.1 |
| re-addition_op_right-002 | addition | multiply | no | 1.0000 | 0.1667 |  | 41.8 |
| re-addition_op_right-003 | addition | add | yes | 1.0000 | 0.2857 |  | 41.3 |
| re-addition_op_right-004 | addition | multiply | no | 0.6667 | 1.0000 |  | 40.9 |
| re-bare_digit_grid-000 | bare_digits | bare_digits | yes | 1.0000 | 0.6667 |  | 42.7 |
| re-bare_digit_grid-001 | bare_digits | bare_digits | yes | 1.0000 | 0.7500 |  | 82.0 |
| re-bare_digit_grid-002 | bare_digits | bare_digits | yes | 1.0000 | 0.7500 |  | 42.5 |
| re-bare_digit_grid-003 | bare_digits | bare_digits | yes | 1.0000 | 0.8333 |  | 43.4 |
| re-bare_digit_grid-004 | bare_digits | bare_digits | yes | 1.0000 | 0.7500 |  | 49.1 |
| re-bare_digit_grid-005 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 42.8 |
| re-bare_digit_grid-006 | bare_digits | bare_digits | yes | 1.0000 | 0.5000 |  | 40.1 |
| re-bare_digit_grid-007 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 42.5 |
| re-bare_digit_grid-008 | bare_digits | bare_digits | yes | 1.0000 | 0.1667 |  | 44.8 |
| re-bare_digit_grid-009 | bare_digits | bare_digits | yes | 1.0000 | 0.8333 |  | 43.2 |
| re-bare_digits-000 | bare_digits | bare_digits | yes | 0.3333 | 1.0000 |  | 42.7 |
| re-bare_digits-001 | bare_digits | bare_digits | yes | 0.3333 | 1.0000 |  | 41.0 |
| re-bare_digits-002 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 40.7 |
| re-bare_digits-003 | bare_digits | bare_digits | yes | 0.6667 | 1.0000 |  | 42.5 |
| re-bare_digits-004 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.2 |
| re-bare_digits-005 | bare_digits | bare_digits | yes | 0.5000 | 1.0000 |  | 42.3 |
| re-bare_digits-006 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 40.8 |
| re-bare_digits-007 | bare_digits | bare_digits | yes | 0.5000 | 1.0000 |  | 44.2 |
| re-bare_digits-008 | bare_digits | bare_digits | yes | 0.5000 | 1.0000 |  | 41.7 |
| re-bare_digits-009 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.4 |
| re-bare_digits-010 | bare_digits | bare_digits | yes | 0.3333 | 1.0000 |  | 41.3 |
| re-bare_digits-011 | bare_digits | bare_digits | yes | 0.3333 | 1.0000 |  | 42.6 |
| re-division-long-000 | division | multiply | no | 0.0625 | 0.0625 |  | 51.7 |
| re-division-long-001 | division | divide | yes | 0.4815 | 0.4444 |  | 91.8 |
| re-division-long-002 | division | divide | yes | 0.0625 | 0.1875 |  | 48.7 |
| re-division-long-003 | division | divide | yes | 0.4444 | 0.2222 |  | 48.9 |
| re-division-long-004 | division | divide | yes | 0.7778 | 0.5556 |  | 44.6 |
| re-division-long-005 | division | multiply | no | 0.2703 | 0.0541 |  | 97.1 |
| re-division-long-006 | division | divide | yes | 0.3333 | 0.2083 |  | 126.9 |
| re-division-long-007 | division | multiply | no | 0.0968 | 0.0968 |  | 124.3 |
| re-division-long-008 | division | divide | yes | 0.1053 | 0.0526 |  | 89.7 |
| re-division-long-009 | division | divide | yes | 0.1111 | 0.7778 |  | 44.3 |
| re-division-long-010 | division | divide | yes | 0.4483 | 0.2069 |  | 52.4 |
| re-division-short-000 | division | bare_digits | no | 1.0000 | 0.8000 |  | 43.7 |
| re-division-short-001 | division | divide | yes | 0.6000 | 1.0000 |  | 42.6 |
| re-division-short-002 | division | divide | yes | 1.0000 | 0.2857 |  | 44.6 |
| re-division-short-003 | division | multiply | no | 1.0000 | 1.0000 |  | 43.9 |
| re-division-short-004 | division | divide | yes | 0.2500 | 0.7500 |  | 45.6 |
| re-division-short-005 | division | divide | yes | 0.8333 | 0.5000 |  | 45.0 |
| re-division-short-006 | division | bare_digits | no | 1.0000 | 1.0000 |  | 42.9 |
| re-division-short-007 | division | bare_digits | no | 1.0000 | 0.5000 |  | 43.2 |
| re-division-short-008 | division | multiply | no | 0.6000 | 0.4000 |  | 44.0 |
| re-division-short-009 | division | divide | yes | 0.3333 | 0.6667 |  | 43.2 |
| re-division-short-010 | division | multiply | no | 1.0000 | 0.5714 |  | 44.6 |
| re-division-short-011 | division | bare_digits | no | 1.0000 | 1.0000 |  | 42.5 |
| re-division-short-012 | division | divide | yes | 0.8000 | 0.4000 |  | 42.6 |
| re-division-short-013 | division | divide | yes | 0.5714 | 1.0000 |  | 44.3 |
| re-division-short-014 | division | divide | yes | 0.3333 | 1.0000 |  | 46.0 |
| re-division-short-015 | division | divide | yes | 0.6667 | 0.5000 |  | 45.0 |
| re-division-short-016 | division | multiply | no | 1.0000 | 0.1429 |  | 117.8 |
| re-division-simple-000 | division | multiply | no | 1.0000 | 1.0000 |  | 44.2 |
| re-division-simple-001 | division | subtract | no | 1.0000 | 1.0000 |  | 117.0 |
| re-division-simple-002 | division | bare_digits | no | 1.0000 | 0.4000 |  | 43.4 |
| re-division-simple-003 | division | subtract | no | 1.0000 | 1.0000 |  | 84.5 |
| re-division-simple-004 | division | divide | yes | 1.0000 | 1.0000 |  | 43.5 |
| re-division-simple-005 | division | divide | yes | 1.0000 | 1.0000 |  | 42.5 |
| re-division-simple-006 | division | subtract | no | 1.0000 | 1.0000 |  | 141.3 |
| re-division-simple-007 | division | divide | yes | 1.0000 | 1.0000 |  | 43.0 |
| re-multiplication-multi-000 | multiplication | multiply | yes | 0.3077 | 0.1923 |  | 49.7 |
| re-multiplication-multi-001 | multiplication | multiply | yes | 0.3478 | 1.0000 |  | 86.0 |
| re-multiplication-multi-002 | multiplication | multiply | yes | 0.3043 | 0.9565 |  | 47.6 |
| re-multiplication-multi-003 | multiplication | subtract | no | 1.0000 | 1.0000 |  | 43.9 |
| re-multiplication-multi-004 | multiplication | add | no | 0.5714 | 1.0000 |  | 121.9 |
| re-multiplication-multi-005 | multiplication | multiply | yes | 0.0857 | 0.2286 |  | 126.9 |
| re-multiplication-multi-006 | multiplication | multiply | yes | 0.2273 | 1.0000 |  | 86.9 |
| re-multiplication-multi-007 | multiplication | multiply | yes | 0.2414 | 1.0000 |  | 159.1 |
| re-multiplication-multi-008 | multiplication | multiply | yes | 1.0000 | 0.1176 |  | 47.2 |
| re-multiplication-multi-009 | multiplication | subtract | no | 1.0000 | 1.0000 |  | 44.1 |
| re-multiplication-multi-010 | multiplication | multiply | yes | 0.2000 | 0.3333 |  | 90.6 |
| re-multiplication-multi-011 | multiplication | multiply | yes | 0.2500 | 0.3500 |  | 48.3 |
| re-multiplication-multi-012 | multiplication | multiply | yes | 0.3000 | 0.4000 |  | 46.7 |
| re-multiplication-multi-013 | multiplication | multiply | yes | 1.0000 | 0.4545 |  | 48.4 |
| re-multiplication-simple-000 | multiplication | add | no | 1.0000 | 1.0000 |  | 43.1 |
| re-multiplication-simple-001 | multiplication | multiply | yes | 1.0000 | 0.6000 |  | 43.2 |
| re-multiplication-simple-002 | multiplication | multiply | yes | 1.0000 | 1.0000 |  | 45.5 |
| re-multiplication-simple-003 | multiplication | multiply | yes | 1.0000 | 1.0000 |  | 43.7 |
| re-multiplication-simple-004 | multiplication | divide | no | 1.0000 | 1.0000 |  | 41.5 |
| re-multiplication-simple-005 | multiplication | multiply | yes | 1.0000 | 1.0000 |  | 42.8 |
| re-multiplication-simple-006 | multiplication | subtract | no | 1.0000 | 0.5000 |  | 43.9 |
| re-multiplication-simple-007 | multiplication | multiply | yes | 1.0000 | 1.0000 |  | 43.1 |
| re-multiplication-simple-008 | multiplication | multiply | yes | 1.0000 | 1.0000 |  | 43.4 |
| re-multiplication-simple-009 | multiplication | unknown | no | 1.0000 | 1.0000 | high_entropy_eq_type | 41.5 |
| re-multiplication-simple-010 | multiplication | add | no | 1.0000 | 0.7500 |  | 43.6 |
| re-multiplication-simple-011 | multiplication | add | no | 0.6667 | 0.5000 |  | 82.5 |
| re-multiplication-simple-012 | multiplication | multiply | yes | 1.0000 | 0.5000 |  | 43.2 |
| re-multiplication-simple-013 | multiplication | multiply | yes | 1.0000 | 0.5000 |  | 87.4 |
| re-multiplication_simple_no_bar-000 | multiplication | unknown | no | 1.0000 | 1.0000 | high_entropy_eq_type | 45.8 |
| re-multiplication_simple_no_bar-001 | multiplication | divide | no | 1.0000 | 1.0000 |  | 44.4 |
| re-multiplication_simple_no_bar-002 | multiplication | multiply | yes | 1.0000 | 1.0000 |  | 44.3 |
| re-multiplication_simple_no_bar-003 | multiplication | bare_digits | no | 0.5000 | 1.0000 |  | 43.1 |
| re-multiplication_simple_op_right-000 | multiplication | multiply | yes | 1.0000 | 1.0000 |  | 42.7 |
| re-multiplication_simple_op_right-001 | multiplication | multiply | yes | 1.0000 | 1.0000 |  | 43.1 |
| re-multiplication_simple_op_right-002 | multiplication | multiply | yes | 1.0000 | 1.0000 |  | 42.9 |
| re-multiplication_simple_op_right-003 | multiplication | unknown | no | 1.0000 | 1.0000 | high_entropy_eq_type | 41.9 |
| re-standalone_bar-000 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.1 |
| re-standalone_bar-001 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.8 |
| re-standalone_bar-002 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.8 |
| re-standalone_bar-003 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 40.6 |
| re-standalone_bar-004 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 81.2 |
| re-standalone_bar-005 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 42.5 |
| re-standalone_bar-006 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 40.5 |
| re-standalone_bar-007 | bare_digits | divide | no | 1.0000 | 0.0000 |  | 80.3 |
| re-standalone_bar-008 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.7 |
| re-standalone_bar-009 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.5 |
| re-standalone_bar-010 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.1 |
| re-standalone_bar-011 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 40.4 |
| re-standalone_bracket-000 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 42.4 |
| re-standalone_bracket-001 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 45.6 |
| re-standalone_bracket-002 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 44.4 |
| re-standalone_bracket-003 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 43.5 |
| re-standalone_bracket-004 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 42.4 |
| re-standalone_bracket-005 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 43.6 |
| re-standalone_bracket-006 | bare_digits | divide | no | n/a | n/a |  | 43.5 |
| re-standalone_bracket-007 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 42.5 |
| re-standalone_bracket-008 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.6 |
| re-standalone_bracket-009 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.7 |
| re-standalone_bracket-010 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.5 |
| re-standalone_bracket-011 | bare_digits | bare_digits | yes | 1.0000 | 1.0000 |  | 41.4 |
| re-subtraction-000 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 46.2 |
| re-subtraction-001 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 45.5 |
| re-subtraction-002 | subtraction | subtract | yes | 0.8000 | 1.0000 |  | 45.0 |
| re-subtraction-003 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 43.0 |
| re-subtraction-004 | subtraction | multiply | no | 1.0000 | 0.4000 |  | 44.0 |
| re-subtraction-005 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 43.7 |
| re-subtraction-006 | subtraction | multiply | no | 1.0000 | 0.4286 |  | 43.3 |
| re-subtraction-007 | subtraction | multiply | no | 1.0000 | 0.4286 |  | 44.6 |
| re-subtraction-008 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 42.9 |
| re-subtraction-009 | subtraction | divide | no | 0.8000 | 0.8000 |  | 43.6 |
| re-subtraction-010 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 43.7 |
| re-subtraction-011 | subtraction | multiply | no | 1.0000 | 0.3333 |  | 45.1 |
| re-subtraction-012 | subtraction | multiply | no | 1.0000 | 0.5556 |  | 85.4 |
| re-subtraction-013 | subtraction | subtract | yes | 1.0000 | 0.8182 |  | 45.0 |
| re-subtraction-014 | subtraction | multiply | no | 1.0000 | 0.5000 |  | 45.8 |
| re-subtraction-015 | subtraction | multiply | no | 0.4286 | 0.4286 |  | 48.5 |
| re-subtraction_heavy_borrow-000 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 48.8 |
| re-subtraction_heavy_borrow-001 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 47.5 |
| re-subtraction_heavy_borrow-002 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 47.4 |
| re-subtraction_heavy_borrow-003 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 45.4 |
| re-subtraction_heavy_borrow-004 | subtraction | multiply | no | 1.0000 | 0.8333 |  | 44.1 |
| re-subtraction_heavy_borrow-005 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 46.0 |
| re-subtraction_heavy_borrow-006 | subtraction | subtract | yes | 1.0000 | 1.0000 |  | 87.0 |
| re-subtraction_heavy_borrow-007 | subtraction | subtract | yes | 1.0000 | 1.0000 |  | 46.5 |
| re-subtraction_heavy_borrow-008 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 44.7 |
| re-subtraction_heavy_borrow-009 | subtraction | add | no | 1.0000 | 1.0000 |  | 42.6 |
| re-subtraction_no_bar-000 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 42.5 |
| re-subtraction_no_bar-001 | subtraction | multiply | no | 0.6667 | 1.0000 |  | 42.0 |
| re-subtraction_no_bar-002 | subtraction | bare_digits | no | 1.0000 | 1.0000 |  | 42.0 |
| re-subtraction_no_bar-003 | subtraction | bare_digits | no | 0.5000 | 1.0000 |  | 41.9 |
| re-subtraction_op_right-000 | subtraction | multiply | no | 1.0000 | 0.1667 |  | 43.2 |
| re-subtraction_op_right-001 | subtraction | multiply | no | 1.0000 | 1.0000 |  | 43.6 |
| re-subtraction_op_right-002 | subtraction | multiply | no | 1.0000 | 0.1667 |  | 43.0 |
| re-subtraction_op_right-003 | subtraction | add | no | 1.0000 | 1.0000 |  | 41.4 |
| re-subtraction_op_right-004 | subtraction | multiply | no | 0.6667 | 1.0000 |  | 42.2 |
| re-subtraction_op_right-005 | subtraction | multiply | no | 1.0000 | 0.6364 |  | 43.0 |

---
Generated: 2026-06-22T14:25:13.258605+00:00
Python: 3.12.13 (main, Mar  3 2026, 12:39:30) [Clang 21.0.0 (clang-2100.0.123.102)]
PyTorch: 2.11.0
Ultralytics: 8.4.54
YOLO SHA: 256482ec
GNN SHA: 482d82f8
