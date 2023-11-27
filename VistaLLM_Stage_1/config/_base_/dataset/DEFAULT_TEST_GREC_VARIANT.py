GREC_TEST_COMMON_CFG = dict(
    type='GRECDataset',
    template_file=r'GREC_ChatGPT.json',
    image_folder=r'train2014',
    max_dynamic_size=None,
)

DEFAULT_TEST_GREC_VARIANT = dict(
    GREC_Val=dict(
        **GREC_TEST_COMMON_CFG,
        filename=r'GREC_ref3_val.jsonl',
    ),
    GREC_TestA=dict(
        **GREC_TEST_COMMON_CFG,
        filename=r'GREC_ref3_testA.jsonl',
    ),
    GREC_TestB=dict(
        **GREC_TEST_COMMON_CFG,
        filename=r'GREC_ref3_testB.jsonl',
    ),
)
