from keras import Model, Sequential
from keras.layers import Concatenate, Dropout, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D, \
    Dense, Flatten, K
from keras.regularizers import L1L2


def create_model_with_branch(embedding_layers, model_descriptor:str):
    "sub_conv[2,3,4](dropout=0.2,conv1d=100-v,)"
    submod_str_start=model_descriptor.index("sub_conv")
    submod_str_end=model_descriptor.index(")")
    submod_str=model_descriptor[submod_str_start: submod_str_end]

    kernel_str=submod_str[submod_str.index("[")+1: submod_str.index("]")]
    dilation_rates=[]
    if "{" in submod_str:
        dilation_str=submod_str[submod_str.index("{")+1:submod_str.index("}")]
        dilation_rates=dilation_str.split(",")
    skipgrams=[]
    if "<" in submod_str: #skipconv1d
        skipgram_str=submod_str[submod_str.index("<")+1:submod_str.index(">")]
        skipgrams=skipgram_str.split(",")
    submod_layer_descriptor = submod_str[submod_str.index("(")+1:]
    submodels = []
    for ks in kernel_str.split(","):
        submodels.append(create_submodel(embedding_layers, submod_layer_descriptor, ks))

    for dr in dilation_rates:
        for ks in kernel_str.split(","):
            submodels.append(create_submodel(embedding_layers, submod_layer_descriptor, ks, dr))

    for sk in skipgrams:
        for ks in kernel_str.split(","):
            skipconv_submodels=(
                create_submodel_with_skipconv1d(embedding_layers, submod_layer_descriptor, int(ks),int(sk)))
            for sm in skipconv_submodels:
                submodels.append(sm)

    submodel_outputs = [model.output for model in submodels]
    if len(submodel_outputs)>1:
        x = Concatenate(axis=1)(submodel_outputs)
    else:
        x=submodel_outputs[0]

    parallel_layers=Model(inputs=embedding_layers[0].input, outputs=x)
    #print("submodel:")
    #parallel_layers.summary()
    #print("\n")

    outter_model_descriptor=model_descriptor[model_descriptor.index(")")+2:]
    big_model = Sequential()
    big_model.add(parallel_layers)
    for layer_descriptor in outter_model_descriptor.split(","):
        ld=layer_descriptor.split("=")

        layer_name=ld[0]
        params=None
        if len(ld)>1:
            params=ld[1].split("-")

        if layer_name=="dropout":
            big_model.add(Dropout(float(params[0])))
        elif layer_name=="lstm":
            if params[1]=="True":
                return_seq=True
            else:
                return_seq=False
            big_model.add(LSTM(units=int(params[0]), return_sequences=return_seq))
        elif layer_name=="gru":
            if params[1]=="True":
                return_seq=True
            else:
                return_seq=False
            big_model.add(GRU(units=int(params[0]), return_sequences=return_seq))
        elif layer_name=="bilstm":
            if params[1]=="True":
                return_seq=True
            else:
                return_seq=False
            big_model.add(Bidirectional(LSTM(units=int(params[0]), return_sequences=return_seq)))
        elif layer_name=="conv1d":
            if len(params)==2:
                big_model.add(Conv1D(filters=int(params[0]),
                             kernel_size=int(params[1]), padding='same', activation='relu'))
            elif len(params)==3:
                print("dilated cnn")
                big_model.add(Conv1D(filters=int(params[0]),
                             kernel_size=int(params[1]), dilation_rate=int(params[2]),padding='same', activation='relu'))
        elif layer_name=="maxpooling1d":
            big_model.add(MaxPooling1D(pool_size=int(params[0])))
        elif layer_name=="gmaxpooling1d":
            big_model.add(GlobalMaxPooling1D())
        elif layer_name == "dense":
            if len(params) == 2:
                big_model.add(Dense(int(params[0]), activation=params[1]))
            elif len(params) > 2:
                kernel_reg = create_regularizer(params[2])
                activity_reg = create_regularizer(params[3])
                if kernel_reg is not None and activity_reg is None:
                    big_model.add(Dense(int(params[0]), activation=params[1],
                                        kernel_regularizer=kernel_reg))
                elif activity_reg is not None and kernel_reg is None:
                    big_model.add(Dense(int(params[0]), activation=params[1],
                                        activity_regularizer=activity_reg))
                elif activity_reg is not None and kernel_reg is not None:
                    big_model.add(Dense(int(params[0]), activation=params[1],
                                        activity_regularizer=activity_reg,
                                        kernel_regularizer=kernel_reg))
        elif layer_name=="flatten":
            big_model.add(Flatten())

    big_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #big_model.summary()

    return big_model


def create_submodel(embedding_layers, submod_layer_descriptor, cnn_ks, cnn_dilation=None):
    model = Sequential()
    if len(embedding_layers)==1:
        model.add(embedding_layers[0])
    else:
        concat_embedding_layers(embedding_layers, model)
    for layer_descriptor in submod_layer_descriptor.split(","):
        if "=" not in layer_descriptor:
            continue
        ld=layer_descriptor.split("=")

        layer_name=ld[0]
        params=None
        if len(ld)>1:
            params=ld[1].split("-")

        if layer_name=="dropout":
            model.add(Dropout(float(params[0])))
        elif layer_name=="lstm":
            if params[1]=="True":
                return_seq=True
            else:
                return_seq=False
            model.add(LSTM(units=int(params[0]), return_sequences=return_seq))
        elif layer_name=="gru":
            if params[1]=="True":
                return_seq=True
            else:
                return_seq=False
            model.add(GRU(units=int(params[0]), return_sequences=return_seq))
        elif layer_name=="bilstm":
            if params[1]=="True":
                return_seq=True
            else:
                return_seq=False
            model.add(Bidirectional(LSTM(units=int(params[0]), return_sequences=return_seq)))
        elif layer_name=="conv1d":
            if cnn_dilation is None:
                model.add(Conv1D(filters=int(params[0]),
                             kernel_size=int(cnn_ks), padding='same', activation='relu'))
            else:
                model.add(Conv1D(filters=int(params[0]),
                             kernel_size=int(cnn_ks), dilation_rate=int(cnn_dilation),
                                 padding='same', activation='relu'))

        elif layer_name=="maxpooling1d":
            size=params[0]
            if size=="v":
                size=int(cnn_ks)
            else:
                size=int(params[0])
            model.add(MaxPooling1D(pool_size=size))
        elif layer_name=="gmaxpooling1d":
            model.add(GlobalMaxPooling1D())
        elif layer_name=="dense":
            model.add(Dense(int(params[0]), activation=params[1]))
    return model


def generate_ks_and_masks(target_cnn_ks, skip):
    masks=[]
    real_cnn_ks=target_cnn_ks+skip
    for gap_index in range(1, real_cnn_ks):
        mask=[]
        for ones in range(0,gap_index):
            mask.append(1)
        for zeros in range(gap_index,gap_index+skip):
            if zeros<real_cnn_ks:
                mask.append(0)
        for ones in range(gap_index+skip, real_cnn_ks):
            if ones <real_cnn_ks:
                mask.append(1)

        if mask[len(mask)-1]!=0:
            masks.append(mask)
    return [real_cnn_ks,masks]


def create_submodel_with_skipconv1d(embedding_layer, submod_layer_descriptor, target_cnn_ks, skip
                                    ):
    submodels=[]
    ks_and_masks=generate_ks_and_masks(target_cnn_ks, skip)
    for mask in ks_and_masks[1]:
        model = Sequential()
        model.add(embedding_layer)
        for layer_descriptor in submod_layer_descriptor.split(","):
            if layer_descriptor.endswith("_"):
                continue
            ld=layer_descriptor.split("=")

            layer_name=ld[0]
            params=None
            if len(ld)>1:
                params=ld[1].split("-")

            if layer_name=="dropout":
                model.add(Dropout(float(params[0])))
            elif layer_name=="lstm":
                if params[1]=="True":
                    return_seq=True
                else:
                    return_seq=False
                model.add(LSTM(units=int(params[0]), return_sequences=return_seq))
            elif layer_name=="gru":
                if params[1]=="True":
                    return_seq=True
                else:
                    return_seq=False
                model.add(GRU(units=int(params[0]), return_sequences=return_seq))
            elif layer_name=="bilstm":
                if params[1]=="True":
                    return_seq=True
                else:
                    return_seq=False
                model.add(Bidirectional(LSTM(units=int(params[0]), return_sequences=return_seq)))
            elif layer_name=="conv1d":
                model.add(SkipConv1D(filters=int(params[0]),
                          kernel_size=int(ks_and_masks[0]), validGrams=mask,
                          padding='same', activation='relu'))

            elif layer_name=="maxpooling1d":
                size=params[0]
                if size=="v":
                    size=int(ks_and_masks[0])
                else:
                    size=int(params[0])
                model.add(MaxPooling1D(pool_size=size))
            elif layer_name=="gmaxpooling1d":
                model.add(GlobalMaxPooling1D())
            elif layer_name=="dense":
                model.add(Dense(int(params[0]), activation=params[1]))
        submodels.append(model)
    return submodels

def concat_embedding_layers(embedding_layers, big_model):
    submodels = []

    for el in embedding_layers:
        m = Sequential()
        m.add(el)
        submodels.append(m)

    submodel_outputs = [model.output for model in submodels]
    if len(submodel_outputs) > 1:
        x = Concatenate(axis=2)(submodel_outputs)
    else:
        x = submodel_outputs[0]

    parallel_layers = Model(inputs=[embedding_layers[0].input, embedding_layers[1].input], outputs=x)
    big_model.add(parallel_layers)


def create_regularizer(string):
    if string=="none":
        return None
    string_array=string.split("_")
    return L1L2(float(string_array[0]),float(string_array[1]))


#a 1D convolution that skips some entries
class SkipConv1D(Conv1D):

    #in the init, let's just add a parameter to tell which grams to skip
    def __init__(self, validGrams, **kwargs):

        #for this example, I'm assuming validGrams is a list
        #it should contain zeros and ones, where 0's go on the skip positions
        #example: [1,1,0,1] will skip the third gram in the window of 4 grams
        assert len(validGrams) == kwargs.get('kernel_size')
        self.validGrams = K.reshape(K.constant(validGrams),(len(validGrams),1,1))
            #the chosen shape matches the dimensions of the kernel
            #the first dimension is the kernel size, the others are input and ouptut channels


        #initialize the regular conv layer:
        super(SkipConv1D,self).__init__(**kwargs)

        #here, the filters, size, etc, go inside kwargs, so you should use them named
        #but you may make them explicit in this __init__ definition
        #if you think it's more comfortable to use it like this


    #in the build method, let's replace the original kernel:
    def build(self, input_shape):

        #build as the original layer:
        super(SkipConv1D,self).build(input_shape)

        #replace the kernel
        self.originalKernel = self.kernel
        self.kernel = self.validGrams * self.originalKernel

