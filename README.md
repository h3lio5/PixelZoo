# PixelZoo
Implementation of various pixel-based autoregressive models like pixelCNN, GatedPixelCNN, ConditionalPixelCNN, PixelCNN++, and PixelSNAIL.

## 1. Setup Instructions and Dependencies
You may setup the repository on your local machine by either downloading it or running the following line on `terminal`.
``` Batchfile
git clone https://github.com/h3lio5/PixelZoo.git
```
All dependencies required by this repo can be downloaded by creating a virtual environment with Python 3.7 and running

``` Batchfile
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install -e .
```
## 2. Training Instructions



## 3. Results
### PixelCNN 
 * With a categorical distribution over 255 pixel values in the last layer, the model appears to perform much better but takes a little longer to train.
 
