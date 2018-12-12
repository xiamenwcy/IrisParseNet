The detailed iris localization evaluation process is listed as follows:
1. use your method to generate iris inner/outer circle/ellipse parameters (transform the results as .ini format, refer to gen_circle_param_ini.m and struct2ini.m)
2. generate iris inner/outer boundary mask according to iris circle/ellipse parameter. Please refer to  gen_circle_mask.hdev or gen_ellipse_mask.hdev
3. transform the mask above to binary mask ,please refer to mask_2_bn.m
4. enter into shape_similariry folder, then run error_h.m
5. enter into shape_sim_curves folder, then run evalution_run_test.m