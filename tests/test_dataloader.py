from rfdnet import SRDataLoader


def test_dataloader():
    dataset = SRDataLoader(
        'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip', batch_size=1
    ).make_dataset()
    x, y = next(iter(dataset))
    assert y.shape[1] == x.shape[1] * 3
    assert y.shape[2] == x.shape[2] * 3
