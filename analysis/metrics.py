# This handles everything following model produces a probability vector for each index ([0-999]) 
# translates probabilities into shape/texture/neither predictions
# computes shape bias per model
# computes per category and per texture bias 
#   meaning which categories are more resistant to texture swaping and thus they have shape-bias
#   and which textures result in a texture-bias meaning they are dominant at ruining CNN predictions
# confidence analysis -> how confident modes are when making shape-based decisions vs texture-based
# all 5 output figures  
