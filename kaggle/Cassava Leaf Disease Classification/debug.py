import timm
model=timm.create_model('seresnet152d', pretrained=False)#nfnet_f4
#print(model)
cnt=0
for i,p in enumerate(model.parameters()):
    #
    cnt+=1
print(cnt)
#print(timm.list_models())
# '''
# ['adv_inception_v3', 'cspdarknet53', 'cspdarknet53_iabn', 'cspresnet50', 'cspresnet50d', 'cspresnet50w', 
# 'cspresnext50', 'cspresnext50_iabn', 'darknet53', 'densenet121', 'densenet121d', 'densenet161', 'densenet169', 
# 'densenet201', 'densenet264', 'densenet264d_iabn', 'densenetblur121d', 'dla34', 'dla46_c', 'dla46x_c', 'dla60', 
# 'dla60_res2net', 'dla60_res2next', 'dla60x', 'dla60x_c', 'dla102', 'dla102x', 'dla102x2', 'dla169', 'dpn68', 'dpn68b', 
# 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'eca_vovnet39b', 'ecaresnet18', 'ecaresnet50', 'ecaresnet50d', 'ecaresnet50d_pruned',
#  'ecaresnet101d', 'ecaresnet101d_pruned', 'ecaresnetlight', 'ecaresnext26tn_32x4d', 'efficientnet_b0', 'efficientnet_b1', 
#  'efficientnet_b1_pruned', 'efficientnet_b2', 'efficientnet_b2_pruned', 'efficientnet_b2a', 'efficientnet_b3', 'efficientnet_b3_pruned',
#   'efficientnet_b3a', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8', 'efficientnet_cc_b0_4e', 
#   'efficientnet_cc_b0_8e', 'efficientnet_cc_b1_8e', 'efficientnet_el', 'efficientnet_em', 'efficientnet_es', 'efficientnet_l2', 'efficientnet_lite0', 
#   'efficientnet_lite1', 'efficientnet_lite2', 'efficientnet_lite3', 'efficientnet_lite4', 'ens_adv_inception_resnet_v2', 'ese_vovnet19b_dw', 
#   'ese_vovnet19b_slim', 'ese_vovnet19b_slim_dw', 'ese_vovnet39b', 'ese_vovnet39b_evos', 'ese_vovnet57b', 'ese_vovnet99b', 'ese_vovnet99b_iabn', 
#   'fbnetc_100', 'gluon_inception_v3', 'gluon_resnet18_v1b', 'gluon_resnet34_v1b', 'gluon_resnet50_v1b', 'gluon_resnet50_v1c', 'gluon_resnet50_v1d',
#    'gluon_resnet50_v1s', 'gluon_resnet101_v1b', 'gluon_resnet101_v1c', 'gluon_resnet101_v1d', 'gluon_resnet101_v1s', 'gluon_resnet152_v1b', 
#    'gluon_resnet152_v1c', 'gluon_resnet152_v1d', 'gluon_resnet152_v1s', 'gluon_resnext50_32x4d', 'gluon_resnext101_32x4d', 'gluon_resnext101_64x4d',
#     'gluon_senet154', 'gluon_seresnext50_32x4d', 'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d', 'gluon_xception65', 'hrnet_w18', 
#     'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40', 'hrnet_w44', 'hrnet_w48', 'hrnet_w64', 
#     'ig_resnext101_32x8d', 'ig_resnext101_32x16d', 'ig_resnext101_32x32d', 'ig_resnext101_32x48d', 'inception_resnet_v2', 'inception_v3', 
#     'inception_v4', 'legacy_senet154', 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50', 'legacy_seresnet101', 'legacy_seresnet152',
#      'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d', 'mixnet_l', 'mixnet_m', 'mixnet_s', 'mixnet_xl', 'mixnet_xxl', 
#      'mnasnet_050', 'mnasnet_075', 'mnasnet_100', 'mnasnet_140', 'mnasnet_a1', 'mnasnet_b1', 'mnasnet_small', 'mobilenetv2_100', 'mobilenetv2_110d', 
#      'mobilenetv2_120d', 'mobilenetv2_140', 'mobilenetv3_large_075', 'mobilenetv3_large_100', 'mobilenetv3_rw', 'mobilenetv3_small_075', 
#      'mobilenetv3_small_100', 'nasnetalarge', 'pnasnet5large', 'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 
#      'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320', 'regnety_002', 'regnety_004', 
#      'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032', 'regnety_040', 'regnety_064', 'regnety_080', 'regnety_120', 'regnety_160', 
#      'regnety_320', 'res2net50_14w_8s', 'res2net50_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s', 'res2net50_48w_2s', 'res2net101_26w_4s',
#       'res2next50', 'resnest14d', 'resnest26d', 'resnest50d', 'resnest50d_1s4x24d', 'resnest50d_4s2x40d', 'resnest101e', 'resnest200e', 'resnest269e', 
#       'resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet34', 'resnet34d', 'resnet50', 'resnet50d', 'resnet101', 'resnet101d', 'resnet101d_320', 
#       'resnet152', 'resnet152d', 'resnet152d_320', 'resnet200', 'resnet200d', 'resnet200d_320', 'resnetblur18', 'resnetblur50', 'resnext50_32x4d', 
#       'resnext50d_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200', 
#       'rexnetr_100', 'rexnetr_130', 'rexnetr_150', 'rexnetr_200', 'selecsls42', 'selecsls42b', 'selecsls60', 'selecsls60b', 'selecsls84', 'semnasnet_050',
#        'semnasnet_075', 'semnasnet_100', 'semnasnet_140', 'senet154', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet50tn', 'seresnet101', 
#        'seresnet152', 'seresnet152d', 'seresnet152d_320', 'seresnext26_32x4d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext26tn_32x4d', 
#        'seresnext50_32x4d', 'seresnext101_32x4d', 'seresnext101_32x8d', 'skresnet18', 'skresnet34', 'skresnet50', 'skresnet50d', 'skresnext50_32x4d', 
#        'spnasnet_100', 'ssl_resnet18', 'ssl_resnet50', 'ssl_resnext50_32x4d', 'ssl_resnext101_32x4d', 'ssl_resnext101_32x8d', 'ssl_resnext101_32x16d', 
#        'swsl_resnet18', 'swsl_resnet50', 'swsl_resnext50_32x4d', 'swsl_resnext101_32x4d', 'swsl_resnext101_32x8d', 'swsl_resnext101_32x16d',
#         'tf_efficientnet_b0', 'tf_efficientnet_b0_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1', 'tf_efficientnet_b1_ap', 'tf_efficientnet_b1_ns', 
#         'tf_efficientnet_b2', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b3', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b3_ns', 
#         'tf_efficientnet_b4', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b5_ns', 
#         'tf_efficientnet_b6', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b6_ns', 'tf_efficientnet_b7', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b7_ns', 
#         'tf_efficientnet_b8', 'tf_efficientnet_b8_ap', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_el',
#          'tf_efficientnet_em', 'tf_efficientnet_es', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 
#          'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'tf_inception_v3', 'tf_mixnet_l', 'tf_mixnet_m', 'tf_mixnet_s', 
#          'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100', 
#          'tf_mobilenetv3_small_minimal_100', 'tresnet_l', 'tresnet_l_448', 'tresnet_m', 'tresnet_m_448', 'tresnet_xl', 'tresnet_xl_448', 'tv_densenet121', 
#          'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152', 'tv_resnext50_32x4d', 'vit_base_patch16_224', 'vit_base_patch16_384', 
#          'vit_base_patch32_384', 'vit_base_resnet26d_224', 'vit_base_resnet50d_224', 'vit_huge_patch16_224', 'vit_huge_patch32_384',
#           'vit_large_patch16_224', 'vit_large_patch16_384', 'vit_large_patch32_384', 'vit_small_patch16_224', 'vit_small_resnet26d_224', 
# 'vit_small_resnet50d_s3_224', 'vovnet39a', 'vovnet57a', 'wide_resnet50_2', 'wide_resnet101_2', 'xception', 'xception41', 'xception65', 'xception71']
# '''