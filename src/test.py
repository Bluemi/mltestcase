from utils import describe, cosine_transform_2d, inv_cosine_transform_2d
from utils.datasets import load_data



def main():
    dataset = load_data('mnist', train=False, num_workers=0, use_dataloader=False)
    img, label = next(iter(dataset))
    img = img.reshape(1, *img.shape)
    # img = torch.rand(1, 28, 28)
    # normed_img = img - torch.mean(img)
    result = cosine_transform_2d(img)
    # describe(result[:, 0, :], 'result first column')
    # describe(result[:, :, 0], 'result first row')
    # result += torch.rand_like(result) * 0.1
    back = inv_cosine_transform_2d(result)

    diff = img - back

    describe(img, 'img')
    describe(result, 'result')

    describe(diff, 'diff')


if __name__ == '__main__':
    main()
