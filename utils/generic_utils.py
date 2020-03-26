import os
import re
import glob
import shutil
import datetime
import json
import subprocess
import importlib
import pickle
import numpy as np
from collections import OrderedDict, Counter

import tensorflow as tf


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config


def get_git_branch():
    try:
        out = subprocess.check_output(["git", "branch"]).decode("utf8")
        current = next(line for line in out.split("\n")
                       if line.startswith("*"))
        current.replace("* ", "")
    except subprocess.CalledProcessError:
        current = "inside_docker"
    return current


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    # try:
    #     subprocess.check_output(['git', 'diff-index', '--quiet',
    #                              'HEAD'])  # Verify client is clean
    # except:
    #     raise RuntimeError(
    #         " !! Commit before training to get the commit hash.")
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    # Not copying .git folder into docker container
    except subprocess.CalledProcessError:
        commit = "0000000"
    print(' > Git Hash: {}'.format(commit))
    return commit


def save_checkpoint(model, optimizer, optimizer_st, model_loss, out_path,
                    current_step, epoch):
    checkpoint_path = 'checkpoint_{}.pkl'.format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print("   | > Checkpoint saving : {}".format(checkpoint_path))
    state = {
        'model': model.get_weights(),
        'optimizer': optimizer,
        'step': current_step,
        'epoch': epoch,
        'linear_loss': model_loss,
        'date': datetime.date.today().strftime("%B %d, %Y"),
        'r': model.decoder.r
    }
    pickle.dump(state, open(checkpoint_path, 'wb'))


def save_best_model(model, optimizer, model_loss, best_loss, out_path,
                    current_step, epoch):
    if model_loss < best_loss:
        state = {
            'model': model.get_weights(),
            'optimizer': optimizer,
            'step': current_step,
            'epoch': epoch,
            'linear_loss': model_loss,
            'date': datetime.date.today().strftime("%B %d, %Y"),
            'r': model.decoder.r
        }
        best_loss = model_loss
        bestmodel_path = 'best_model.pkl'
        bestmodel_path = os.path.join(out_path, bestmodel_path)
        print("\n > BEST MODEL ({0:.5f}) : {1:}".format(
            model_loss, bestmodel_path))
        pickle.dump(state, open(bestmodel_path, 'wb'))
    return best_loss


def create_experiment_folder(root_path, model_name, debug):
    """ Create a folder with the current date and time """
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    # if debug:
    # commit_hash = 'debug'
    # else:
    commit_hash = get_commit_hash()
    output_folder = os.path.join(
        root_path, model_name + '-' + date_str + '-' + commit_hash)
    os.makedirs(output_folder, exist_ok=True)
    print(" > Experiment folder: {}".format(output_folder))
    return output_folder


def remove_experiment_folder(experiment_path):
    """Check folder if there is a checkpoint, otherwise remove the folder"""

    checkpoint_files = glob.glob(experiment_path + "/*.pth.tar")
    if not checkpoint_files:
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
            print(" ! Run is removed from {}".format(experiment_path))
    else:
        print(" ! Run is kept in {}".format(experiment_path))


def copy_config_file(config_file, out_path, new_fields):
    config_lines = open(config_file, "r").readlines()
    # add extra information fields
    for key, value in new_fields.items():
        if type(value) == str:
            new_line = '"{}":"{}",\n'.format(key, value)
        else:
            new_line = '"{}":{},\n'.format(key, value)
        config_lines.insert(1, new_line)
    config_out_file = open(out_path, "w")
    config_out_file.writelines(config_lines)
    config_out_file.close()


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.max()
    batch_size = sequence_length.size(0)
    seq_range = np.empty([0, max_len], dtype=np.int8)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (
        sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    # B x T_max
    return seq_range_expand < seq_length_expand


# @tf.custom_gradient
def check_gradient(x, grad_clip):
    x_normed = tf.clip_by_norm(x, grad_clip)
    grad_norm = tf.norm(grad_clip)
    return x_normed, grad_norm


def count_parameters(model, c):
    try:
        return model.count_params()
    except:
        input_dummy = tf.convert_to_tensor(np.random.rand(8, 128).astype('int32'))
        input_lengths = np.random.randint(100, 129, (8, ))
        input_lengths[-1] = 128
        input_lengths = tf.convert_to_tensor(input_lengths.astype('int32'))
        mel_spec = np.random.rand(8, 2 * c.r,
                                  c.audio['num_mels']).astype('float32')
        mel_spec = tf.convert_to_tensor(mel_spec)
        speaker_ids = np.random.randint(
            0, 5, (8, )) if c.use_speaker_embedding else None
        _ = model(input_dummy, input_lengths, mel_spec)
        return model.count_params()


def split_dataset(items):
    is_multi_speaker = False
    speakers = [item[-1] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    eval_split_size = 500 if len(items) * 0.01 > 500 else int(
        len(items) * 0.01)
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        # most stupid code ever -- Fix it !
        while len(items_eval) < eval_split_size:
            speakers = [item[-1] for item in items]
            speaker_counter = Counter(speakers)
            item_idx = np.random.randint(0, len(items))
            if speaker_counter[items[item_idx][-1]] > 1:
                items_eval.append(items[item_idx])
                del items[item_idx]
        return items_eval, items
    else:
        return items[:eval_split_size], items[eval_split_size:]


def gradual_training_scheduler(global_step, config):
    new_values = None
    for values in config.gradual_training:
        if global_step >= values[0]:
            new_values = values
    return new_values[1], new_values[2]


class KeepAverage():
    def __init__(self):
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        return self.avg_values[key]

    def add_value(self, name, init_val=0, init_iter=0):
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if weighted_avg:
            self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
            self.iters[name] += 1
        else:
            self.avg_values[name] = self.avg_values[name] * \
                self.iters[name] + value
            self.iters[name] += 1
            self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)


def setup_model(num_chars, num_speakers, c):
    print(" > Using model: {}".format(c.model))
    MyModel = importlib.import_module('TTS_tf.models.' + c.model.lower())
    MyModel = getattr(MyModel, c.model)
    if c.model.lower() in "tacotron":
        model = MyModel(num_chars=num_chars,
                        num_speakers=num_speakers,
                        r=c.r,
                        linear_dim=1025,
                        mel_dim=80,
                        gst=c.use_gst,
                        memory_size=c.memory_size,
                        attn_win=c.windowing,
                        attn_norm=c.attention_norm,
                        prenet_type=c.prenet_type,
                        prenet_dropout=c.prenet_dropout,
                        use_forward_attn=c.use_forward_attn,
                        use_trans_agent=c.transition_agent,
                        use_forward_attn_mask=c.forward_attn_mask,
                        use_loc_attn=c.location_attn,
                        separate_stopnet=c.separate_stopnet)
    # elif c.model.lower() == "tacotron2":
    #     model = MyModel(num_chars=num_chars,
    #                     num_speakers=num_speakers,
    #                     r=c.r,
    #                     attn_win=c.windowing,
    #                     attn_norm=c.attention_norm,
    #                     prenet_type=c.prenet_type,
    #                     prenet_dropout=c.prenet_dropout,
    #                     forward_attn=c.use_forward_attn,
    #                     trans_agent=c.transition_agent,
    #                     forward_attn_mask=c.forward_attn_mask,
    #                     location_attn=c.location_attn,
    #                     separate_stopnet=c.separate_stopnet)
    return model
