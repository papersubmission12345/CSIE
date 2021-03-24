This repository is the implementation of CSIE-M: Deep learning-based compressive sensing image enhancement using multiple reconstructed signals

## Requirements
- Python 3 (Anaconda is recommended)
- skimage
- imageio
- Pytorch (Pytorch version >=0.4.1 is recommended)
- tqdm 
- pandas
- cv2 (pip install opencv-python)
- Matlab 

##Test
To calculate the quality score of CS reconstructed image:
cd /CSIE-M/Scorenet/
python test.py -opt ./options/test/testScorenet.json
To enhance the images by MRRN:
cd /CSIE/CSIE-3/
python test.py -opt ./options/test/testMRRN.json
Note: Please set the path in json file to your project directory.