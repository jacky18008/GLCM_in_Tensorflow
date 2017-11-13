
# coding: utf-8

# In[27]:

from keras.layers import Layer
from keras import backend as K
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np


# In[50]:

# It should be a class the need to build it as a keras layer.
class GLCM(Layer):
    # batch_size is absolutely needed, and the data overlap should be hold by user.
    def __init__(self, batch_size = 32, distance = 1, angle = 0, level = 2, symmetric = True, normed = True, **kwargs ):
        super(GLCM, self).__init__(**kwargs)
        self.distance = distance
        self.angle = angle
        self.level = level
        self.symmetric = symmetric
        self.normed = normed
        self.batch_size = batch_size
    
    
    # all the inputs are assumed as 2D images like (batch, row, column, channel)
    # The main function of this layer.
    def call(self, inputs):
        
        #cast the images into fixed channels and int32.
        self.image = tf.cast(inputs/(3/self.level), dtype=tf.int32)
        
        #some necessary parameters
        channel = inputs.shape[3].value
        rows = self.image.shape[1].value
        cols = self.image.shape[2].value
        channel = self.image.shape[3].value

        #calculate the subarray size of image 
        row = int(round(np.sin(self.angle))) * self.distance
        col = int(round(np.cos(self.angle))) * self.distance

        #We have two subarrays with same shape to express the two distinct entries
        #We wil use Matrix operations instead of for-loop and if, for the ban on them from tensorflow. 
        if col > 0:
            self.subarray_1 = self.image[:, :rows-row, :cols-col, :]
            self.subarray_2 = self.image[:, row:, col:, :]
        else:
            self.subarray_1 = self.image[:, :self.rows-row, -col:, :]
            self.subarray_2 = self.image[:, row:, :self.cols+col, :]
         
        # The sub_rows and sub_columns of our sub_arrays.
        sub_row = self.subarray_1.shape[1].value
        sub_column = self.subarray_1.shape[2].value
    
        # For tensorflow can only tile for dim<5, we will do like: "tile->reshape" most of time.
        reshaped_subarray_1 = tf.reshape(self.subarray_1, shape=(self.batch_size, sub_row, sub_column, channel, 1))
        reshaped_subarray_2 = tf.reshape(self.subarray_2, shape=(self.batch_size, sub_row, sub_column, channel, 1))
        
        tiled_subarray_1 = tf.tile(reshaped_subarray_1, [1, 1, 1, 1, self.level*self.level*(self.level-1)])
        tiled_subarray_2 = tf.tile(reshaped_subarray_2, [1, 1, 1, 1, self.level*self.level*(self.level-1)])

        reshape_tiled_subarray_1 = tf.reshape(tiled_subarray_1, shape=(self.batch_size, sub_row, sub_column, channel, self.level, self.level, self.level-1))
        reshape_tiled_subarray_2 = tf.reshape(tiled_subarray_2, shape=(self.batch_size, sub_row, sub_column, channel, self.level, self.level, self.level-1))
        
        #This part we initial the constant sequence that will be used on computing.
        #We do some operations to make it to match the shape we want.
        sequence = tf.constant([])

        for i in range(self.level):
            hold = tf.ones(shape=(self.level-1))*i
            sequence = tf.concat([sequence, hold], axis=0)

        sequence = tf.cast(sequence, tf.int32)

        sequence_reshape = tf.reshape(sequence, (self.level-1, self.level))
        sequence_transpose = tf.transpose(sequence_reshape)
        sequence_rev = tf.reverse(sequence_transpose, [0])
        
        sequence_rev_reshape = tf.reshape(sequence_rev, (1, 1, 1, 1, 1, self.level*(self.level-1)))
        sequence_rev_reshape_tile = tf.tile(sequence_rev_reshape, [self.batch_size, sub_row, sub_column, channel, self.level, 1])
        sequence_rev_reshaped_tile = tf.reshape(sequence_rev_reshape_tile, (self.batch_size, sub_row, sub_column, channel, self.level, self.level, self.level-1))

        #Lagrange polynomials Denominator
        GLCM_t = tf.tile(tf.reshape(sequence, (1, 1, 1, 1, self.level*(self.level-1))), [self.batch_size, sub_row, sub_column, channel, self.level])
        GLCM_t_reshape = tf.reshape(GLCM_t, (self.batch_size, sub_row, sub_column, channel, self.level, self.level, self.level-1))
        GLCM_Denominator_array = GLCM_t_reshape-sequence_rev_reshaped_tile
        GLCM_Denominator = tf.reduce_prod(GLCM_Denominator_array, axis=6)
        
        #Lagrange polynomials Numerator
        GLCM_Numerator_1_array = reshape_tiled_subarray_1-sequence_rev_reshaped_tile
        GLCM_Numerator_2_array = reshape_tiled_subarray_2-sequence_rev_reshaped_tile

        GLCM_Numerator_1 = tf.reduce_prod(GLCM_Numerator_1_array, axis=6)
        GLCM_Numerator_2 = tf.reduce_prod(GLCM_Numerator_2_array, axis=6)
        
        # Subarrays generated after Lagrange polynomials.
        GLCM_subarray_1 = GLCM_Numerator_1/GLCM_Denominator
        GLCM_subarray_2 = GLCM_Numerator_2/GLCM_Denominator
        
        # Now, we need to do logic checking and merging on our subarrays to get the final GLCM.

        # a and b = a*b
        # We have to do transpose on GLCM_subarray_2 for its different location with GLCM_subarray_1 in GLCM. 
        GLCM_subarray_2_transpose = tf.transpose(GLCM_subarray_2, perm=[0, 1, 2, 3, 5, 4])
        GLCM_single_entry = tf.multiply(GLCM_subarray_1, GLCM_subarray_2_transpose)
        
        # sum single entries to get the final GLCM.
        GLCM = tf.reduce_sum(GLCM_single_entry, axis=[1, 2])

        if self.symmetric:
            GLCM = GLCM+tf.transpose(GLCM, perm=[0, 1, 3, 2])
        if self.normed:
            GLCM = GLCM/tf.reduce_sum(GLCM)
        
        #For some unknown casting by tensorflow function(?), we have to cast it back to float32 somehow.
        #Same reason to reshape.
        GLCM = tf.reshape(GLCM, (self.batch_size, channel*self.level*self.level))
        GLCM = tf.cast(GLCM, tf.float32)
        
            
        return GLCM
    
    #This function would be called in model compiling.
    def compute_output_shape(self, input_shape):
        #(batch_size, channel, GLCM_row, GLCM_column)
        return (self.batch_size, input_shape[3]*self.level*self.level)
    
    


# In[ ]:



