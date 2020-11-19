import tensorflow as tf
import tqdm

from model.parametric_model import Model
from input_producer import IP
from utils import *

ORDER = 4
LOSS = 'mse'
IS_CNN = False

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    model_name = ''
    model_name += 'parametric_' + LOSS
    model_name = model_name + '_cnn/' if IS_CNN else model_name + '_mlp/'
    print(model_name)
    img_path = os.path.join('imgs', model_name)
    mkdir(img_path)

    model = Model(ORDER+1, LOSS, IS_CNN)

    train_producer = IP(k_regression=False).init_producer(batch_per_class=100)
    test_producer = IP(k_regression=False, is_train=False, num_split=1).init_producer(batch_per_class=100)

    print('model name: ', model_name)
    writer = tf.summary.create_file_writer(os.path.join('runs', model_name))
    for step in tqdm.tqdm(range(10000)):
        batch_train, batch_target = train_producer()

        loss = model.train_on_batch(batch_train, batch_target)
        if step%100==0:
            batch_test, batch_target = test_producer()
            test_loss = model.get_loss(batch_test, batch_target)
            # print('step: {}, train loss: {:.6f}, test loss: {:.6f}'.format(step, loss, test_loss))

            # save_figs(batch_test, batch_target, model, img_path, step, regression=K_REGRESSION)
            # save_figs(batch_test, batch_target, model, img_path, step, converting=False, regression=K_REGRESSION)
            converted = save_figs_to_arr(batch_test, batch_target, model, regression=False)[:, :, :3]
            converted = tf.convert_to_tensor(converted)
            signal = save_figs_to_arr(batch_test, batch_target, model, converting=False, regression=False)[:, :, :3]
            signal = tf.convert_to_tensor(signal)

            with writer.as_default():
                tf.summary.scalar('loss/test_loss', test_loss, step=step)
                tf.summary.scalar('loss_logscale/test_loss', np.log(test_loss), step=step)
                tf.summary.image('converted', [converted], step=step)
                tf.summary.image('signal', [signal], step=step)

        _, params = model(batch_train, batch_target, return_params=True)
        with writer.as_default():
            tf.summary.scalar('loss/loss', loss, step=step)
            tf.summary.scalar('loss_logscale/loss', np.log(loss), step=step)
            tf.summary.histogram('params', params, step=step)
