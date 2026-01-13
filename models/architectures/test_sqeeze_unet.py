import matplotlib.pyplot as plt
import numpy as np

from Sqeeze_Unet import *

if __name__ == "__main__":
    arg = (dfm.input_length, dfm.num_classes, dfm.deconv_ksize, dfm.model_width, dfm.dropout, dfm.Conv_activation, dfm.activation)
    unet_model = SqeezeUnet1D(*arg).SqueezeUNet()

    handle = AIProcess(unet_model, work_dir=dfm.WORKSPACE)
    handle.compile_model()
    handle.train(epochs=50, batch_size=10)

    # unet_model.load_weights(tf.train.latest_checkpoint('/mnt/ai_data/PPG2ABP_data_second/model_unet/checkpoint')).expect_partial()
    # unet_model.save('/mnt/ai_data/PPG2ABP_data_second/model_unet/unet_v{}.h5'.format(1))

    # ppg, abp = [], []
    #
    # tf_data = AIData(dfm.DIR_SAVE_TFRECORD, 256)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_2/AI_data_0.tfrecord', batch_size=1)
    # from tqdm import tqdm
    # for index, batch in tqdm(enumerate(ds)):
    #     list_tensors = [i for i in batch]
    #     ppg.extend([tensor.numpy().flatten() for tensor in list_tensors[0]])
    #     abp.extend([tensor.numpy().flatten() for tensor in list_tensors[1]])
    #     _ppg = np.asarray(ppg[index]).reshape(1, len(ppg[index]), 1)
    #     abp_predict = unet_model.predict(_ppg)
    #     plt.plot(ppg[index], color='b')
    #     plt.plot(abp_predict.flatten(), color='r')
    #     plt.plot(abp[index], color='k')
    #     plt.show()
    #     plt.close()
