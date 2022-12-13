import torch
from torchvision import transforms, models
import numpy as np
import copy
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset import load_binary_imagenet_test_data, BinaryImageNet


class Model(object):

    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.model.to(device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def classify_one(self, x):
        assert len(x.shape) == 3
        x = torch.round(x) / 255.0
        x = self.normalize(x)
        x = x.unsqueeze(0)
        #return torch.argmax(self.model(x), dim=-1)[0]
        #print(torch.argmax(self.model(x), dim=-1)[0], torch.argmax(self.model(x), dim=-1)[0].item() in BinaryImageNet.DOG_LABELS)
        return int(torch.argmax(self.model(x), dim=-1)[0].item() in BinaryImageNet.DOG_LABELS)


def l2_distance(a, b):
    if type(b) != torch.Tensor:
        return (torch.round(a) / 255.0).norm()

    dist = (torch.round(a) / 255.0 - torch.round(b) / 255.0).norm()
    return dist


def value_mask_init(patch_num):
    value_mask = torch.ones([patch_num, patch_num]).cuda()
    return value_mask


def noise_mask_init(x, image, patch_num, patch_size):
    noise = x - image
    noise_mask = torch.zeros([patch_num, patch_num]).cuda()
    for row_counter in range(patch_num):
        for col_counter in range(patch_num):
            noise_mask[row_counter][col_counter] = l2_distance(
                noise[:, (row_counter * patch_size):(row_counter * patch_size + patch_size),
                      (col_counter * patch_size):(col_counter * patch_size + patch_size)], 0)
    return noise_mask


def translate(index, patch_num):
    best_row = index // patch_num
    best_col = index - patch_num * best_row

    return best_row, best_col


class PatchAttack(object):

    def __init__(self, model):
        self.model = model

    def patch_attack(self, original, label, starting_point, iterations=1000, binary_search=False):

        step = 0
        noises = []
        bad_queries = 0
        bad_queries_from_binary_search = 0
        num_binary_searches = 0

        patch_num = 4
        patch_size = int(original.shape[1] / patch_num)

        success_num = 0
        fail_num = 0

        value_mask = value_mask_init(patch_num)
        noise_mask = noise_mask_init(starting_point, original, patch_num, patch_size)

        # print(value_mask)
        # print(noise_mask)
        # print(torch.sum(value_mask * noise_mask))

        best_noise = starting_point - original
        current_min_noise = l2_distance(starting_point, original)
        print("min", current_min_noise)

        while step < iterations:
            if torch.sum(value_mask * noise_mask) == 0:
                print("patch num * 2", step)
                print("min", current_min_noise)
                patch_num *= 2

                if patch_num == 64:
                    print("only", step)
                    break

                patch_size = int(original.shape[1] / patch_num)
                value_mask = value_mask_init(patch_num)
                noise_mask = noise_mask_init(best_noise, original, patch_num, patch_size)

            total_mask = value_mask * noise_mask
            best_index = torch.argmax(total_mask)
            best_row, best_col = translate(best_index, patch_num)

            temp_noise = copy.deepcopy(best_noise)

            temp_noise[:, (best_row * patch_size):(best_row * patch_size + patch_size),
                       (best_col * patch_size):(best_col * patch_size + patch_size)] = 0

            candidate = torch.clip(torch.round(original + temp_noise), 0, 255)

            if l2_distance(candidate, original) >= current_min_noise:
                assert l2_distance(candidate, original) == current_min_noise
                # print("not worth", torch.sum(value_mask).item())
                value_mask[best_row, best_col] = 0
                continue

            temp_result = self.model.classify_one(candidate)
            step += 1

            is_adversarial = (temp_result != label)
            bad_queries += int(~is_adversarial)

            if is_adversarial:
                print(step, current_min_noise.item(), l2_distance(candidate, original).item(), "Success")
                current_min_noise = l2_distance(candidate, original)
                success_num += 1
                best_noise = candidate - original
                noise_mask[best_row, best_col] = l2_distance(
                    best_noise[:, (best_row * patch_size):(best_row * patch_size + patch_size),
                               (best_col * patch_size):(best_col * patch_size + patch_size)], 0)
                noises.append((bad_queries, current_min_noise.item()))

            else:
                print(step, current_min_noise.item(), l2_distance(candidate, original).item(), "Fail")
                fail_num += 1

                noises.append((bad_queries, current_min_noise.item()))

                if binary_search:
                    num_binary_searches += 1
                    good_noise = copy.deepcopy(best_noise)
                    bad_noise = temp_noise

                    while torch.abs(torch.max(good_noise - bad_noise)) > 1:
                        mid_noise = (good_noise + bad_noise) / 2
                        candidate = torch.clip(torch.round(original + mid_noise), 0, 255)
                        mid_result = self.model.classify_one(candidate)
                        step += 1

                        is_mid_adversarial = (mid_result != label)

                        bad_queries += int(~is_mid_adversarial)
                        bad_queries_from_binary_search += int(~is_mid_adversarial)

                        if is_mid_adversarial:
                            good_noise = mid_noise
                        else:
                            bad_noise = mid_noise

                        current_min_noise = l2_distance(good_noise, 0)
                        noises.append((bad_queries, current_min_noise.item()))
                        print("binary search", step,
                              torch.abs(torch.max(good_noise - bad_noise)).item(),
                              l2_distance(candidate, original).item())

                    best_noise = good_noise
                    noise_mask[best_row, best_col] = l2_distance(
                        best_noise[:, (best_row * patch_size):(best_row * patch_size + patch_size),
                                   (best_col * patch_size):(best_col * patch_size + patch_size)], 0)
                    candidate = torch.clip(torch.round(original + best_noise), 0, 255)
                    print(step, l2_distance(candidate, original).item(), "Success")
                    current_min_noise = l2_distance(candidate, original)

                # don't consider this patch anymore
                value_mask[best_row, best_col] = 0

        final_best_adv_example = best_noise + original
        final_best_adv_example = final_best_adv_example

        print("success_num", success_num, fail_num)
        return final_best_adv_example, step, noises, bad_queries, bad_queries_from_binary_search, num_binary_searches


device = "cuda"

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# torch.backends.cudnn.deterministic = True
# random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# np.random.seed(1)
# data = ImageNet(root="/data/imagenet", split='val', transform=preprocess)
# data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True, num_workers=0)
# X, Y = next(iter(data_loader))

data_loader = load_binary_imagenet_test_data(test_batch_size=1000)
X, Y = next(iter(data_loader))

X *= 255.0
X = X.to(device)
Y = Y.to(device)

print(X.shape, Y.shape)


def find_original_noise(model, x, y, inverted=False):
    """
    std = 0.01
    while True:
        noise = torch.round(torch.randn_like(x) * std)
        candidate = torch.round(torch.clip(x + noise, 0, 255))
        if model.classify_one(candidate) != y:
            break

        std *= 2

    print("std", std)
    print("noise", (noise/255.0).norm())
    return candidate
    """
    epsilons = np.linspace(0, 1, num=100 + 1)[1:]
    queries = 0

    if inverted:
        epsilons = epsilons[::-1]
    prev = None

    for epsilon in epsilons:
        min_, max_ = 0, 255.0
        std = epsilon / np.sqrt(3) * (max_ - min_)
        noise = torch.randn_like(x) * std

        perturbed = x + epsilon * noise
        perturbed = torch.round(torch.clamp(perturbed, min_, max_))
        queries += 1
        if inverted:
            if model.classify_one(perturbed) == y:
                return prev, queries
            prev = torch.clone(perturbed)
        else:
            if model.classify_one(perturbed) != y:
                return perturbed, queries

    assert False


model = Model()

binary_search = bool(int(sys.argv[1]))

all_orig_queries = []
all_noises = []
all_bad_queries = []
all_bad_queries_from_binary_search = []
all_num_binary_searches = []

k = 0

for i in range(len(X)):
    x = X[i]
    y = Y[i]

    if y.item() != 1 or model.classify_one(x) != y:
        continue

    torch.manual_seed(i)
    candidate, orig_queries = find_original_noise(model, x, y, inverted=False)
    all_orig_queries.append(orig_queries)

    adv, step, noises, bad_queries, bad_queries_from_binary_search, num_binary_searches = \
        PatchAttack(model).patch_attack(x, y, candidate, iterations=10000, binary_search=binary_search)
    print(l2_distance(adv, x))
    assert model.classify_one(adv) != y
    print(noises)
    print("bad_queries", bad_queries)
    print("bad_queries_from_binary_search", bad_queries_from_binary_search)
    print("num_binary_searches", num_binary_searches)

    if (not binary_search) and (k < 10):
        plt.imshow(x.cpu().permute(1, 2, 0).numpy().astype(np.uint8))
        plt.savefig("x_{}.png".format(len(all_noises)))
        plt.imshow(adv.cpu().permute(1, 2, 0).numpy().astype(np.uint8))
        plt.savefig("adv_{}.png".format(len(all_noises)))

    all_bad_queries.append(bad_queries)
    all_bad_queries_from_binary_search.append(bad_queries_from_binary_search)
    all_num_binary_searches.append(num_binary_searches)

    noise_extend = np.zeros((10000, 2))
    noise_extend[:len(noises)] = noises
    noise_extend[len(noises):] = noises[-1]
    all_noises.append(noise_extend)
    k += 1

all_noises = np.array(all_noises)
np.save(f"all_noises_{binary_search}.npy", all_noises)
print(all_noises.shape)
all_noises_med = np.median(all_noises[:, :, 1], axis=0)
print(list(all_noises_med))
all_noises_avg = np.mean(all_noises[:, :, 1], axis=0)
print(list(all_noises_avg))

print("orig_queries (avg)", np.mean(all_orig_queries))
print("orig_queries (med)", np.median(all_orig_queries))

print("bad_queries (avg)", np.mean(all_bad_queries))
print("bad_queries_from_binary_search (avg)", np.mean(all_bad_queries_from_binary_search))
print("num_binary_searches (avg)", np.mean(all_num_binary_searches))

print("bad_queries (median)", np.median(all_bad_queries))
print("bad_queries_from_binary_search (median)", np.median(all_bad_queries_from_binary_search))
print("num_binary_searches (median)", np.median(all_num_binary_searches))
