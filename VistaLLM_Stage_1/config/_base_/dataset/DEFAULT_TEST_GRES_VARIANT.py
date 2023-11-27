GRES_TEST_COMMON_CFG = dict(
    type='GRESDataset',
    template_file=r'GRES_ChatGPT.json',
    image_folder=r'train2014',
    max_dynamic_size=None,
)

DEFAULT_TEST_GRES_VARIANT = dict(
    GRES_Val=dict(
        **GRES_TEST_COMMON_CFG,
        filename=r'GRES_ref3_val_contour.jsonl',
    ),
    GRES_TestA=dict(
        **GRES_TEST_COMMON_CFG,
        filename=r'GRES_ref3_testA_contour.jsonl',
    ),
    GRES_TestB=dict(
        **GRES_TEST_COMMON_CFG,
        filename=r'GRES_ref3_testB_contour.jsonl',
    ),
)
