# PokeMLPrototype

## Building upon musicmilif's pokemon autoencoder by using it to denoise the pokemon images
   We generate the noise by using numpy's random normal distribution function with mean being equal to the mean pixel value of the pokemon and the standard deviation of 0.08. This gave it a kind of salt and pepper like look to the image. Then we took the pixels of the image and fed it to the encoder neural network. The bottleneck layer was then fed to the decoder where attempts were made to reconstruct the original image. 
   We first tried modifying the original repository jupyter notebook by making changes to incorporate noise. THe file can be found name the same as the orignal file.
   
   The pokemon images that we used came from the musicmilif link https://github.com/musicmilif/Pokemon-Generator
In our own version of the notebook, we assumed that the pokemon images were in a folder title Pokemon and used it directly without regex expressions.
