import tensorflow as tf
import tqdm

from model import AE, End2End
from input_producer import IP
from utils import *

ORDER = 4
K_REGRESSION = False
LOSS = 'mse'
IS_CNN = True

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    model_name = 'test_'
    model_name += 'regression_' if K_REGRESSION else 'end2end_'
    model_name += LOSS
    model_name = model_name + '_cnn/' if IS_CNN else model_name + '_mlp/'
    print(model_name)
    img_path = os.path.join('imgs', model_name)
    mkdir(img_path)

    if K_REGRESSION:
        model = AE(ORDER+1, LOSS, IS_CNN)
    else:
        model = End2End(LOSS, IS_CNN)

    train_producer = IP(k_regression=K_REGRESSION).init_producer(batch_per_class=32)
    test_producer = IP(k_regression=K_REGRESSION, is_train=False, num_split=1).init_producer(batch_per_class=32)

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
            converted = save_figs_to_arr(batch_test, batch_target, model, regression=K_REGRESSION)[:, :, :3]
            converted = tf.convert_to_tensor(converted)
            signal = save_figs_to_arr(batch_test, batch_target, model, converting=False, regression=K_REGRESSION)[:, :, :3]
            signal = tf.convert_to_tensor(signal)

            with writer.as_default():
                tf.summary.scalar('loss/test_loss', test_loss, step=step)
                tf.summary.scalar('loss_logscale/test_loss', np.log(test_loss), step=step)
                tf.summary.image('converted', [converted], step=step)
                tf.summary.image('signal', [signal], step=step)

        with writer.as_default():
            tf.summary.scalar('loss/loss', loss, step=step)
            tf.summary.scalar('loss_logscale/loss', np.log(loss), step=step)
