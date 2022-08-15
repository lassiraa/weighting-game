# Directory of validation images and output for new videos 
in="CHANGE"  # e.g. ./coco/images/val2017/
out="CHANGE"  # e.g. ./3d-videos/ remember to create the folder

cd ./3d-ken-burns/
python autozoom.py --in ${in} --out ${out}
