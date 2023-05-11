echo "SignOPT K ImageNet line search"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_72404164acdb498ca8de12a71d3e8dbb /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_dff3a75e4aa547caad8cd9f1ce118842 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_081f5cd310cb4910b99530d7485c8b67 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_7a002c1f76f74780a509424a275e5a5c --names "k = 1" "k = 1.5" "k = 2" "k = 2.5" "k = 3" --out-path /cluster/project/privsec/realistic-adv-examples/plots/sign_opt_imagenet_distance.pdf --unsafe-only --max-queries 1000 --max-samples 500

echo "SignOPT K ImageNet binary search"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2e47eae06f7544fea37e7f982baac03c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_8d0359f995ed44a8b5a9793c8cfb7f5c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2c204b20ca4c462a856b92a56da6257b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_0ef1abc5aeae45aabc97297f2c94fc4c --names "k = 1" "k = 1.5" "k = 2" "k = 2.5" --out-path /cluster/project/privsec/realistic-adv-examples/plots/sign_opt_imagenet_distance_binary.pdf --unsafe-only --max-queries 1000 --max-samples 500

echo "SignOPT K ImageNet tradeoff"
python plot_dist_vs_queries.py tradeoff --max-queries 1000 --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_dff3a75e4aa547caad8cd9f1ce118842 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_081f5cd310cb4910b99530d7485c8b67 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_7a002c1f76f74780a509424a275e5a5c --names "k = 1.5" "k = 2" "k = 2.5" "k = 3" --out-path /cluster/project/privsec/realistic-adv-examples/plots/sign_opt_imagenet_tradeoff.pdf --max-samples 500

echo "ImageNet L2"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_3387c465662343989bd8ac68df0d870c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2e47eae06f7544fea37e7f982baac03c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_e273debbe9af4195bb82d8c611d4e278 --names Boundary OPT "Stealthy OPT" SignOPT "Stealthy SignOPT" "HSJA" --to-simulate 2 4 --unsafe-only --max-queries 1000 --max-samples 500 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_l2.pdf

echo "ImageNet L2 tradeoff"
python plot_dist_vs_queries.py tradeoff --max-queries 1000 --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2c204b20ca4c462a856b92a56da6257b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_e273debbe9af4195bb82d8c611d4e278 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_3387c465662343989bd8ac68df0d870c --to-simulate 0 2 --names "Stealthy OPT" "OPT" "Stealthy SignOPT" "SignOPT" "HSJA" "Boundary" --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_l2_tradeoff.pdf

echo "ImageNet Linf"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_07f64a8002dd4427bb3953634bac6b40 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_3411439d755e4da99db1dca0a6f0f7ae /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.000_6006e667199a4bc9a55120ae761b6a05 --names HSJA RayS "Stealthy RayS" --unsafe-only --max-queries 1000 --max-samples 500 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_linf.pdf

echo "ImageNet Linf tradeoff"
python plot_dist_vs_queries.py tradeoff --max-queries 1000 --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.000_6006e667199a4bc9a55120ae761b6a05 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_3411439d755e4da99db1dca0a6f0f7ae /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_07f64a8002dd4427bb3953634bac6b40 --names "Stealthy RayS" RayS HSJA --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_linf_tradeoff.pdf

echo "Binary ImageNet L2"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_641ad9fce19e4470a28ee458fd4381d9 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_badbb8b53eed450fa56e5476075ba9c4 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_1f31fb519e2f4105976cddd51ae64593 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_ff8b44a563684828a995ca328854bb87 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_acca32297a324a96aec0b2e311de24d8 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_43f77a226f84448497a8bc1642fe5bc6 --names Boundary OPT "Stealthy OPT" SignOPT "Stealthy SignOPT" "HSJA" --to-simulate 2 4 --unsafe-only --max-queries 1000 --max-samples 500 --out-path /cluster/project/privsec/realistic-adv-examples/plots/binary_l2.pdf

echo "Binary ImageNet L2 tradeoff"
python plot_dist_vs_queries.py tradeoff --max-queries 1000 --exp-paths /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_1f31fb519e2f4105976cddd51ae64593 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_badbb8b53eed450fa56e5476075ba9c4 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_acca32297a324a96aec0b2e311de24d8 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_ff8b44a563684828a995ca328854bb87 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_43f77a226f84448497a8bc1642fe5bc6 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_641ad9fce19e4470a28ee458fd4381d9 --names "Stealthy OPT" OPT "Stealthy SignOPT" SignOPT "HSJA" Boundary --to-simulate 0 4 --out-path /cluster/project/privsec/realistic-adv-examples/plots/binary_l2_tradeoff.pdf

echo "Binary ImageNet Linf"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/binary_imagenet/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_88ea4efd0f984d9b98f524cb86a4baf1 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.050_7be236e21303434ca4a9db8fdca05ba9 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.050_e9c606454b2441d4b3ef5bb680e522a9 --names HSJA RayS "Stealthy RayS" --unsafe-only --max-queries 1000 --max-samples 500 --out-path /cluster/project/privsec/realistic-adv-examples/plots/binary_linf.pdf

echo "Binary ImageNet Linf tradeoff"
python plot_dist_vs_queries.py tradeoff --max-queries 1000 --exp-paths /cluster/project/privsec/realistic-adv-examples/binary_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.050_e9c606454b2441d4b3ef5bb680e522a9 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.050_7be236e21303434ca4a9db8fdca05ba9 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_88ea4efd0f984d9b98f524cb86a4baf1 --names "Stealthy RayS" RayS HSJA --out-path /cluster/project/privsec/realistic-adv-examples/plots/binary_linf_tradeoff.pdf

echo "NSFW L2"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_92ca096d96b1497b8a73e1ce940189b3 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_442d37e2c3be41c19c75664fe4c59387 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_line_0.000_6800e8a81ebd45a5873150a90bc13f9c /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_1421c4ae2096462dbb358ae7601970e8 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_3d97b6f234bd4b629d4c82a06515a5e7 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_68edb7c577ec438680d85f984fdcd3a4 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_45f10a51d320456290ff3a7d749b0854 --unsafe-only --max-queries 1000 --names Boundary OPT "Stealthy OPT" SignOPT "Stealthy SignOPT" HSJA "Stealthy OPT (ideal search)" --to-simulate 2 4 --to-simulate-ideal 6 --draw-legend tr --out-path /cluster/project/privsec/realistic-adv-examples/plots/nsfw_l2.pdf --max-samples 100

echo "NSFW L2 tradeoff"
python plot_dist_vs_queries.py tradeoff --max-queries 1000 --max-queries 1000 --exp-paths /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_line_0.000_6800e8a81ebd45a5873150a90bc13f9c /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_442d37e2c3be41c19c75664fe4c59387 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_3d97b6f234bd4b629d4c82a06515a5e7 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_1421c4ae2096462dbb358ae7601970e8 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_68edb7c577ec438680d85f984fdcd3a4 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_92ca096d96b1497b8a73e1ce940189b3 --names "Stealthy OPT" OPT "Stealthy SignOPT" SignOPT HSJA Boundary --to-simulate 0 2 --draw-legend tr --out-path /cluster/project/privsec/realistic-adv-examples/plots/nsfw_l2_tradeoff.pdf --max-samples 100

echo "NSFW Linf"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_1bc9b08063a4461a9a44ed90f380098d /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_4556c34799c143589702c88368c0070e /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/linf/rays/discrete-0_targeted-0_early-0_line_0.000_85db5c0c173847b1917416ac1b0739b5 --names HSJA RayS "Stealthy RayS" --unsafe-only --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/nsfw_linf.pdf --draw-legend tr --max-samples 100

echo "NSFW Linf tradeoff"
python plot_dist_vs_queries.py tradeoff --max-queries 1000 --max-queries 1000 --exp-paths /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/linf/rays/discrete-0_targeted-0_early-0_line_0.000_85db5c0c173847b1917416ac1b0739b5 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_4556c34799c143589702c88368c0070e /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_1bc9b08063a4461a9a44ed90f380098d --names "Stealthy RayS" RayS HSJA --draw-legend tr --out-path /cluster/project/privsec/realistic-adv-examples/plots/nsfw_linf_tradeoff.pdf --max-samples 100

echo "OPT tradeoff ImageNet"
python plot_dist_vs_queries.py tradeoff --max-queries 1000 --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b --names "OPT (line search)" "OPT (2 line searches)" "OPT (binary)" --to-simulate 0 --out-path /cluster/project/privsec/realistic-adv-examples/plots/opt_imagenet_tradeoff.pdf

echo "OPT distances ImageNet"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd --names "OPT (binary)" "OPT (2 line searches)" "OPT (line search)" --to-simulate 2 --out-path /cluster/project/privsec/realistic-adv-examples/plots/opt_imagenet_distance.pdf --max-queries 1000 --unsafe-only

echo "SignOPT distances ImageNet"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2c204b20ca4c462a856b92a56da6257b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc --names "OPT (binary)" "OPT (2 line searches)" "OPT (line search)" --to-simulate 2 --out-path /cluster/project/privsec/realistic-adv-examples/plots/sign_opt_imagenet_distance.pdf --max-queries 1000 --unsafe-only

echo "RayS tradeoff ImageNet"
python plot_dist_vs_queries.py tradeoff --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.000_6006e667199a4bc9a55120ae761b6a05 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.050_b6707760b49342539d8e36a322639331 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_eggs_dropping_0.050_076e526f56914de8b2c8e8105cc8da17 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_3411439d755e4da99db1dca0a6f0f7ae --names "RayS (line search + early stop)" "RayS (line search)" "RayS (2 line searches + early stop)" "RayS (binary)" --out-path /cluster/project/privsec/realistic-adv-examples/plots/rays_imagenet_tradeoff.pdf --max-queries 500

echo "RayS distances ImageNet"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_3411439d755e4da99db1dca0a6f0f7ae /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_eggs_dropping_0.050_076e526f56914de8b2c8e8105cc8da17 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.050_b6707760b49342539d8e36a322639331 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.000_6006e667199a4bc9a55120ae761b6a05 --names "RayS (binary)" "RayS (line search)" "RayS (2 line searches + early stop)" "RayS (line search + early stop)" --out-path /cluster/project/privsec/realistic-adv-examples/plots/rays_imagenet_distance.pdf --max-queries 500 --unsafe-only

echo "OPT beta NSFW"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_442d37e2c3be41c19c75664fe4c59387 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_binary_0.050_623272daceb14e1cbf0014a4454ecd3c --names "beta = 0.01" "beta = 0.001" --unsafe-only --max-queries 5000 --max-samples 50 --out-path /cluster/project/privsec/realistic-adv-examples/plots/ablation_opt_nsfw_beta.pdf --draw-legend y

echo "SignOPT beta NSFW"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_1421c4ae2096462dbb358ae7601970e8 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.050_0334e6ae4a1a44f19e1aff5e6bb97b39 --names "beta = 0.01" "beta = 0.001" --unsafe-only --max-queries 5000 --max-samples 100 --out-path /cluster/project/privsec/realistic-adv-examples/plots/ablation_sign_opt_nsfw_beta.pdf --draw-legend y

echo "OPT beta + steps NSFW"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_line_0.000_6800e8a81ebd45a5873150a90bc13f9c /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_line_0.000_ab313851df104f64a754fa933eeff832 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_line_0.000_5e2eba0309f64ca9896cf4da922530c1 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/opt/discrete-0_targeted-0_early-0_line_0.000_251b24d4a3da4632a2c5a3ff4bf74a04 --names "beta = 0.01, 100 steps" "beta = 0.001, 100 steps" "beta = 0.01, 200 steps" "beta = 0.001, 200 steps" --unsafe-only --max-queries 800 --max-samples 100 --out-path /cluster/project/privsec/realistic-adv-examples/plots/ablation_opt_nsfw_beta_search_steps.pdf --draw-legend y --to-simulate 0 1 2 3

echo "SignOPT beta + steps NSFW"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_7520a1fd94f64ef5b83efc2ade922021 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_6cfbbf62fa514d48993fa97f68c7f9fe /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_3d97b6f234bd4b629d4c82a06515a5e7 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_ecfc7f6af9494f09b08100d540fc1fd6 /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_185bf1633a5541489b92d88e9d058cad /cluster/project/privsec/realistic-adv-examples/imagenet_nsfw/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_255dd08893fe4c5db02245cc4075c07d --names "k = 1.5, beta = 0.01" "k = 2, beta = 0.01" "k = 2.5, beta = 0.01" "k = 1.5, beta = 0.001" "k = 2, beta = 0.001" "k = 2.5, beta = 0.001" --unsafe-only --max-samples 100 --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/ablation_opt_nswf_beta_k.pdf --draw-legend y

echo "Google NSFW 0.4"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/google_cloud_nsfw_0.4/linf/rays/discrete-1_targeted-0_early-0_binary_0.000_bf9cf5b795a445a2a1b30a0955c488e1 /cluster/project/privsec/realistic-adv-examples/google_cloud_nsfw_0.4/linf/rays/discrete-1_targeted-0_early-0_line_0.000_903d0938d7b24c1d813a4c4ad6b60cd2 --names RayS "Stealthy RayS" --unsafe-only --max-queries 500 --max-samples 200 --out-path /cluster/project/privsec/realistic-adv-examples/plots/google_nsfw_0.4.pdf

echo "Google NSFW 0.6"
python plot_dist_vs_queries.py distance --draw-legend y --exp-paths /cluster/project/privsec/realistic-adv-examples/google_cloud_nsfw_0.6/linf/rays/discrete-1_targeted-0_early-0_binary_0.000_fd27e3180b134975815afb8544e9f3ea /cluster/project/privsec/realistic-adv-examples/google_cloud_nsfw_0.6/linf/rays/discrete-1_targeted-0_early-0_line_0.000_c8a2f53325e447a1ae7ea60e59335221 --names RayS "Stealthy RayS" --unsafe-only --max-queries 500 --max-samples 200 --out-path /cluster/project/privsec/realistic-adv-examples/plots/google_nsfw_0.6.pdf

echo "ImageNet L2 overall"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_3387c465662343989bd8ac68df0d870c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2e47eae06f7544fea37e7f982baac03c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_e273debbe9af4195bb82d8c611d4e278 --names Boundary OPT SignOPT HSJA --max-queries 5000 --max-samples 500 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_l2_overall.pdf --draw-legend y

echo "ImageNet Linf overall"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_07f64a8002dd4427bb3953634bac6b40 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_3411439d755e4da99db1dca0a6f0f7ae --names HSJA RayS --max-queries 5000 --max-samples 500 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_linf_overall.pdf --draw-legend y

echo "ImageNet L2 cost plot, 0.001"
python plot_dist_vs_queries.py cost --draw-legend tr --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_3387c465662343989bd8ac68df0d870c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2e47eae06f7544fea37e7f982baac03c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_e273debbe9af4195bb82d8c611d4e278 --names Boundary OPT "Stealthy OPT" SignOPT "Stealthy SignOPT" "HSJA" --to-simulate 2 4 --bad-query-cost 1. --max-samples 500 --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_l2_cost_0.001.pdf --query-cost 0.001

echo "ImageNet L2 cost plot, 0.0001"
python plot_dist_vs_queries.py cost --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_3387c465662343989bd8ac68df0d870c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2e47eae06f7544fea37e7f982baac03c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_e273debbe9af4195bb82d8c611d4e278 --names Boundary OPT "Stealthy OPT" SignOPT "Stealthy SignOPT" "HSJA" --to-simulate 2 4 --bad-query-cost 1. --max-samples 500 --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_l2_cost_0.0001.pdf --query-cost 0.0001

echo "ImageNet L2 cost plot, 0.00001"
python plot_dist_vs_queries.py cost --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_3387c465662343989bd8ac68df0d870c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2e47eae06f7544fea37e7f982baac03c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_e273debbe9af4195bb82d8c611d4e278 --names Boundary OPT "Stealthy OPT" SignOPT "Stealthy SignOPT" "HSJA" --to-simulate 2 4 --bad-query-cost 1. --max-samples 500 --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_l2_cost_0.00001.pdf --query-cost 0.00001

echo "ImageNet Linf cost plot, 0.1"
python plot_dist_vs_queries.py cost --draw-legend tr --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_07f64a8002dd4427bb3953634bac6b40 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_3411439d755e4da99db1dca0a6f0f7ae /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.000_6006e667199a4bc9a55120ae761b6a05 --names HSJA RayS "Stealthy RayS" --bad-query-cost 1. --max-samples 500 --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_linf_cost_0.1.pdf --query-cost 0.1

echo "ImageNet Linf cost plot, 0.01"
python plot_dist_vs_queries.py cost --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_07f64a8002dd4427bb3953634bac6b40 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_3411439d755e4da99db1dca0a6f0f7ae /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.000_6006e667199a4bc9a55120ae761b6a05 --names HSJA RayS "Stealthy RayS" --bad-query-cost 1. --max-samples 500 --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_linf_cost_0.01.pdf --query-cost 0.01

echo "ImageNet Linf cost plot, 0.001"
python plot_dist_vs_queries.py cost --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/hsja/discrete-0_targeted-0_early-0_binary_0.000_07f64a8002dd4427bb3953634bac6b40 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_binary_0.000_3411439d755e4da99db1dca0a6f0f7ae /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/linf/rays/discrete-0_targeted-0_early-0_line_0.000_6006e667199a4bc9a55120ae761b6a05 --names HSJA RayS "Stealthy RayS" --bad-query-cost 1. --max-samples 500 --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_linf_cost_0.001.pdf --query-cost 0.001

echo "ImageNet L2 with ideal OPT"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_3387c465662343989bd8ac68df0d870c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_2e47eae06f7544fea37e7f982baac03c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_1449b6a517f440acaa349245ded473cc /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_e273debbe9af4195bb82d8c611d4e278 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_1a48815e2dbe41ca9364eca191fb6577 --names Boundary OPT "Stealthy OPT" SignOPT "Stealthy SignOPT" "HSJA" "Stealthy OPT (ideal search)" --to-simulate 2 4 --to-simulate-ideal 6 --unsafe-only --max-queries 1000 --max-samples 500 --out-path /cluster/project/privsec/realistic-adv-examples/plots/imagenet_l2_with_ideal_opt.pdf

echo "Binary L2 with ideal OPT"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/boundary/discrete-0_targeted-0_early-0_binary_0.000_641ad9fce19e4470a28ee458fd4381d9 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_badbb8b53eed450fa56e5476075ba9c4 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_1f31fb519e2f4105976cddd51ae64593 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_binary_0.000_ff8b44a563684828a995ca328854bb87 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/sign_opt/discrete-0_targeted-0_early-0_line_0.000_acca32297a324a96aec0b2e311de24d8 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_43f77a226f84448497a8bc1642fe5bc6 /cluster/project/privsec/realistic-adv-examples/binary_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_3028269f94b6480f8c7aaeb43da8206c --names Boundary OPT "Stealthy OPT" SignOPT "Stealthy SignOPT" "HSJA" "Stealthy OPT (ideal search)" --to-simulate 2 4 --to-simulate-ideal 6 --unsafe-only --max-queries 1000 --max-samples 500 --out-path /cluster/project/privsec/realistic-adv-examples/plots/binary_l2_with_ideal_opt.pdf

echo "HSJA gradient estimation"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_e273debbe9af4195bb82d8c611d4e278 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/hsja/discrete-0_targeted-0_early-0_binary_0.000_527a1c38c3824e038d8cab8a84274df5  --names "HSJA" "Stealthy OPT" "HSJA+Stealthy OPT grad estimation" --unsafe-only --max-queries 1000 --max-samples 300 --out-path /cluster/project/privsec/realistic-adv-examples/plots/hsja_grad_estimations.pdf --draw-legend tr --to-simulate-ideal 2 --to-simulate 1

echo "Binary OPT k ablation"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_dfe447538e3648f69f85d25a6fc614a3 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_c1665061ae13417197d8b610b12ca9ff /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_52f9356a94564f54bf436d5ff6e5fb2b /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_004c97bc68544f2e96a060c83166a59c /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_binary_0.000_470d33cc000c4e4e9c643b4979ab7073 --names 5 8 10 15 20 --unsafe-only --draw-legend tr --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/opt_k_ablation_binary.pdf

echo "Stealthy OPT k ablation"
python plot_dist_vs_queries.py distance --exp-paths /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_5990e8dd78084ccb96bbd763a10bb0b8 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_d95e53364685412cb635bc9fa254fa99 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_9df3bfdb95df4f0db64bd834652039fd /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_f7e85aab3d754766aee0d0a4fddf7ed0 /cluster/project/privsec/realistic-adv-examples/resnet_imagenet/l2/opt/discrete-0_targeted-0_early-0_line_0.000_63f76d06334a4e5da42bcab6eafa2ff2 --names 5 8 10 15 20 --unsafe-only --draw-legend tr --max-queries 1000 --out-path /cluster/project/privsec/realistic-adv-examples/plots/opt_k_ablation_line.pdf --to-simulate 0 1 2 3 4
