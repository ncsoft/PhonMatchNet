import sys, os, datetime, warnings, argparse
import tensorflow as tf
import numpy as np

from model import ukws
from dataset import libriphrase, google, qualcomm
from criterion import total
from criterion.utils import eer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.simplefilter("ignore")

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

checkpoint_dir = './interspeech/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
tensorboard_prefix = os.path.join(checkpoint_dir, "tensorboard")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', required=True, type=int)
parser.add_argument('--lr', required=True, type=float)
parser.add_argument('--loss_weight', default=[1.0, 1.0], nargs=2, type=float)
parser.add_argument('--text_input', required=False, type=str, default='g2p_embed')
parser.add_argument('--audio_input', required=False, type=str, default='both')

parser.add_argument('--train_pkl', required=False, type=str, default='/home/DB/LibriPhrase/data/train_both.pkl')
parser.add_argument('--google_pkl', required=False, type=str, default='/home/DB/google_speech_commands/google.pkl')
parser.add_argument('--qualcomm_pkl', required=False, type=str, default='/home/DB/qualcomm_keyword_speech_dataset/qualcomm.pkl')
parser.add_argument('--libriphrase_pkl', required=False, type=str, default='/home/DB/LibriPhrase/data/test_both.pkl')

parser.add_argument('--stack_extractor', action='store_true')

parser.add_argument('--comment', required=False, type=str)
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()

# Batch size per GPU
GLOBAL_BATCH_SIZE = 2048 * strategy.num_replicas_in_sync
BATCH_SIZE_PER_REPLICA = GLOBAL_BATCH_SIZE / strategy.num_replicas_in_sync

# Make Dataloader
text_input = args.text_input
audio_input = args.audio_input

train_dataset = libriphrase.LibriPhraseDataloader(batch_size=GLOBAL_BATCH_SIZE, features=text_input, train=True, types='both', shuffle=True, pkl=args.train_pkl)
test_dataset = libriphrase.LibriPhraseDataloader(batch_size=GLOBAL_BATCH_SIZE, features=text_input, train=False, types='both', shuffle=True, pkl=args.libriphrase_pkl)
test_easy_dataset = libriphrase.LibriPhraseDataloader(batch_size=GLOBAL_BATCH_SIZE, features=text_input, train=False, types='easy', shuffle=True, pkl=args.libriphrase_pkl)
test_hard_dataset = libriphrase.LibriPhraseDataloader(batch_size=GLOBAL_BATCH_SIZE, features=text_input, train=False, types='hard', shuffle=True, pkl=args.libriphrase_pkl)
test_google_dataset = google.GoogleCommandsDataloader(batch_size=GLOBAL_BATCH_SIZE, features=text_input, shuffle=True, pkl=args.google_pkl)
test_qualcomm_dataset = qualcomm.QualcommKeywordSpeechDataloader(batch_size=GLOBAL_BATCH_SIZE, features=text_input, shuffle=True, pkl=args.qualcomm_pkl)

# Number of phonemes
vocab = train_dataset.nPhoneme
  
# Convert tf.utils.sequence to tf.dataset
train_dataset = libriphrase.convert_sequence_to_dataset(train_dataset)
test_dataset = libriphrase.convert_sequence_to_dataset(test_dataset)
test_easy_dataset = libriphrase.convert_sequence_to_dataset(test_easy_dataset)
test_hard_dataset = libriphrase.convert_sequence_to_dataset(test_hard_dataset)
test_google_dataset = google.convert_sequence_to_dataset(test_google_dataset)
test_qualcomm_dataset = qualcomm.convert_sequence_to_dataset(test_qualcomm_dataset)

# Make disribute dataset for multi-gpu
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
test_easy_dist_dataset = strategy.experimental_distribute_dataset(test_easy_dataset)
test_hard_dist_dataset = strategy.experimental_distribute_dataset(test_hard_dataset)
test_google_dist_dataset = strategy.experimental_distribute_dataset(test_google_dataset)
test_qualcomm_dist_dataset = strategy.experimental_distribute_dataset(test_qualcomm_dataset)

# Model params.
kwargs = {
        'vocab' : vocab,
        'text_input' : text_input,
        'audio_input' : audio_input,
        'frame_length' : 400, 
        'hop_length' : 160, 
        'num_mel'  : 40, 
        'sample_rate' : 16000,
        'log_mel' : False,
        'stack_extractor' : args.stack_extractor,
    }

# Train params.
EPOCHS = args.epoch
lr = args.lr

# Make tensorboard dict.
param = kwargs
param['epoch'] = EPOCHS
param['lr'] = lr
param['loss weight'] = args.loss_weight
param['comment'] = args.comment

with strategy.scope():
    loss_object = total.TotalLoss_SCE(weight=args.loss_weight)
       
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_d = tf.keras.metrics.Mean(name='train_loss_Utt')
    train_loss_sce = tf.keras.metrics.Mean(name='train_loss_Phon')
    
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_loss_d = tf.keras.metrics.Mean(name='test_loss_Utt')
    
    train_auc = tf.keras.metrics.AUC(name='train_auc')
    train_eer = eer(name='train_eer')
    
    test_auc = tf.keras.metrics.AUC(name='test_auc')
    test_eer = eer(name='test_eer')
    
    test_easy_auc = tf.keras.metrics.AUC(name='test_easy_auc')
    test_easy_eer = eer(name='test_easy_eer')
    test_hard_auc = tf.keras.metrics.AUC(name='test_hard_auc')
    test_hard_eer = eer(name='test_hard_eer')
    
    google_auc = tf.keras.metrics.AUC(name='google_auc')
    google_eer = eer(name='google_eer')
    qualcomm_auc = tf.keras.metrics.AUC(name='qualcomm_auc')
    qualcomm_eer = eer(name='qualcomm_eer')

    model = ukws.BaseUKWS(**kwargs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    @tf.function
    def train_step(inputs):
        clean_speech, noisy_speech, text, labels, speech_labels, text_labels = inputs
        with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
            model(clean_speech, text, training=False)
            tape.watch(model.trainable_variables)
            prob, affinity_matrix, LD, sce_logit = model(noisy_speech, text, training=True)
            loss, LD, LC = loss_object(labels, LD, speech_labels, text_labels, sce_logit)
            loss /= GLOBAL_BATCH_SIZE
            LC /= GLOBAL_BATCH_SIZE
            LD /= GLOBAL_BATCH_SIZE
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss.update_state(loss)
        train_loss_d.update_state(LD)
        train_loss_sce.update_state(LC)
        train_auc.update_state(labels, prob)
        train_eer.update_state(labels, prob)
        
        return loss, tf.expand_dims(tf.cast(affinity_matrix * 255, tf.uint8), -1), labels

    @tf.function
    def test_step(inputs):
        clean_speech = inputs[0]
        text = inputs[1]
        labels = inputs[2]
        prob, affinity_matrix, LD, LC = model(clean_speech, text, training=False)[:4]
            
        t_loss, LD = total.TotalLoss(weight=args.loss_weight[0])(labels, LD)
        t_loss /= GLOBAL_BATCH_SIZE
        LD /= GLOBAL_BATCH_SIZE
        
        test_loss.update_state(t_loss)
        test_loss_d.update_state(LD)
        test_auc.update_state(labels, prob)
        test_eer.update_state(labels, prob)
        
        return t_loss, tf.expand_dims(tf.cast(affinity_matrix * 255, tf.uint8), -1), labels
    
    @tf.function
    def test_step_metric_only(inputs, metric=[]):
        clean_speech = inputs[0]
        text = inputs[1]
        labels = inputs[2]

        prob = model(clean_speech, text, training=False)[0]
        
        for m in metric:
            m.update_state(labels, prob)

    train_log_dir = os.path.join(tensorboard_prefix, "train")
    test_log_dir = os.path.join(tensorboard_prefix, "test")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def distributed_train_step(dataset_inputs):
        per_replica_losses, per_replica_affinity_matrix, per_replica_labels = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), strategy.experimental_local_results(per_replica_affinity_matrix)[0], strategy.experimental_local_results(per_replica_labels)[0]

    def distributed_test_step(dataset_inputs):
        per_replica_losses, per_replica_affinity_matrix, per_replica_labels = strategy.run(test_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), strategy.experimental_local_results(per_replica_affinity_matrix)[0], strategy.experimental_local_results(per_replica_labels)[0]
    
    def distributed_test_step_metric_only(dataset_inputs, metric=[]):
        strategy.run(test_step_metric_only, args=(dataset_inputs, metric))

    with train_summary_writer.as_default():
            tf.summary.text('Hyperparameters', tf.stack([tf.convert_to_tensor([k, str(v)]) for k, v in param.items()]), step=0)
    
    for epoch in range(EPOCHS):
        # TRAIN LOOP
        train_matrix = None
        train_labels = None
        test_matrix = None
        train_labels = None
        
        for i, x in enumerate(train_dist_dataset):
            _, train_matrix, train_labels = distributed_train_step(x)
        
        match_train_matrix = []
        unmatch_train_matrix = []
        for i, x in enumerate(train_labels):
            if x == 1:
                match_train_matrix.append(train_matrix[i])
            elif x == 0:
                unmatch_train_matrix.append(train_matrix[i])
            
        with train_summary_writer.as_default():
            tf.summary.scalar('0. Total loss', train_loss.result(), step=epoch)
            tf.summary.scalar('1. Utterance-level Detection loss', train_loss_d.result(), step=epoch)
            tf.summary.scalar('2. Phoneme-levle Detection loss', train_loss_sce.result(), step=epoch)
            tf.summary.scalar('3. AUC', train_auc.result(), step=epoch)
            tf.summary.scalar('4. EER', train_eer.result(), step=epoch)
            tf.summary.image("Affinity matrix (match)", match_train_matrix, max_outputs=5, step=epoch)
            tf.summary.image("Affinity matrix (unmatch)", unmatch_train_matrix, max_outputs=5, step=epoch)
            
        # TEST LOOP
        for x in test_dist_dataset:
            _, test_matrix, test_labels = distributed_test_step(x)
        
        match_test_matrix = []
        unmatch_test_matrix = []
        for i, x in enumerate(test_labels):
            if x == 1:
                match_test_matrix.append(test_matrix[i])
            elif x == 0:
                unmatch_test_matrix.append(test_matrix[i])
                
        for x in test_easy_dist_dataset:
            distributed_test_step_metric_only(x, metric=[test_easy_auc, test_easy_eer])
            
        for x in test_hard_dist_dataset:
            distributed_test_step_metric_only(x, metric=[test_hard_auc, test_hard_eer])

        for x in test_google_dist_dataset:
            distributed_test_step_metric_only(x, metric=[google_auc, google_eer])
            
        for x in test_qualcomm_dist_dataset:
            distributed_test_step_metric_only(x, metric=[qualcomm_auc, qualcomm_eer])
            
        with test_summary_writer.as_default():
            tf.summary.scalar('0. Total loss', test_loss.result(), step=epoch)
            tf.summary.scalar('1. Utterance-level Detection loss', test_loss_d.result(), step=epoch)
            tf.summary.scalar('3. AUC', test_auc.result(), step=epoch)
            tf.summary.scalar('3. AUC (EASY)', test_easy_auc.result(), step=epoch)
            tf.summary.scalar('3. AUC (HARD)', test_hard_auc.result(), step=epoch)
            tf.summary.scalar('3. AUC (Google)', google_auc.result(), step=epoch)
            tf.summary.scalar('3. AUC (Qualcomm)', qualcomm_auc.result(), step=epoch)
            tf.summary.scalar('4. EER', test_eer.result(), step=epoch)
            tf.summary.scalar('4. EER (EASY)', test_easy_eer.result(), step=epoch)
            tf.summary.scalar('4. EER (HARD)', test_hard_eer.result(), step=epoch)
            tf.summary.scalar('4. EER (Google)', google_eer.result(), step=epoch)
            tf.summary.scalar('4. EER (Qualcomm)', qualcomm_eer.result(), step=epoch)
            tf.summary.image("Affinity matrix (match)", match_test_matrix, max_outputs=5, step=epoch)
            tf.summary.image("Affinity matrix (unmatch)", unmatch_test_matrix, max_outputs=5, step=epoch)

        if epoch % 1 == 0:
            checkpoint.save(checkpoint_prefix)
        
        template = ("Epoch {} | TRAIN | Loss {:.3f}, AUC {:.2f}, EER {:.2f} | EER | G {:.2f}, Q {:.2f}, LE {:.2f}, LH {:.2f} | AUC | G {:.2f}, Q {:.2f}, LE {:.2f}, LH {:.2f} |")
        print (template.format(epoch + 1, 
                                train_loss.result(),
                                train_auc.result() * 100,
                                train_eer.result() * 100,
                                google_eer.result() * 100,                        
                                qualcomm_eer.result() * 100,
                                test_easy_eer.result() * 100,
                                test_hard_eer.result() * 100,
                                google_auc.result() * 100,
                                qualcomm_auc.result() * 100,
                                test_easy_auc.result() * 100,
                                test_hard_auc.result() * 100,
                            )
               )

        train_loss.reset_states()
        test_loss.reset_states()
        train_auc.reset_states()
        test_auc.reset_states()
        test_easy_auc.reset_states()
        test_hard_auc.reset_states()
        train_eer.reset_states()
        test_eer.reset_states()
        test_easy_eer.reset_states()
        test_hard_eer.reset_states()
        google_eer.reset_states()
        qualcomm_eer.reset_states()
        google_auc.reset_states()
        qualcomm_auc.reset_states()