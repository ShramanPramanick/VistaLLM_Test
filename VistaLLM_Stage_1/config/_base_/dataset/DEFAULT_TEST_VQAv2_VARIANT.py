VQAv2_TEST_COMMON_CFG = dict(
    type='VQAv2Dataset',
    image_folder=r'VQAv2/combined',
    template_file=r"VQA.json",
)

DEFAULT_TEST_VQAv2_VARIANT = dict(
    VQAv2_val=dict(
        **VQAv2_TEST_COMMON_CFG,
        filename=r'v2_OpenEnded_mscoco_val2014_questions.jsonl',
    ),
    VQAv2_testdev=dict(
        **VQAv2_TEST_COMMON_CFG,
        filename=r'v2_OpenEnded_mscoco_test-dev2015_questions.jsonl',
        has_annotation=False,
    ),
    VQAv2_test=dict(
        **VQAv2_TEST_COMMON_CFG,
        filename=r'v2_OpenEnded_mscoco_test2015_questions.jsonl',
        has_annotation=False,
    ),
)
