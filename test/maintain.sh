/bin/rm -rf result/6oim
python modeling.py --cuda --pdb 6oim
python modeling.py --cuda --pdb 6oim -c D
python modeling.py --cuda --pdb 6oim --ref_ligand ./result/6oim/6oim_B_MG.pdb
python modeling.py --cuda --protein ./result/6oim/6oim.pdb --prefix 6oim
python modeling.py --cuda --protein ./result/6oim/6oim.pdb --ref_ligand ./result/6oim/6oim_B_MG.pdb --prefix 6oim
python modeling.py --cuda --protein ./result/6oim/6oim.pdb --center 1.872 -8.260 -1.361 --prefix 6oim
python screening.py -p ./result/6oim/6oim_D_MOV_model.pm -d ./examples/library/ --cpus 4 -o tmp.csv
python feature_extraction.py --cuda -p ./result/6oim/6oim.pdb --center 1.872 -8.260 -1.361 -o tmp.pt
