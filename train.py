import argparse
import os
import sys
import time
import traceback
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(1)

from TTS_tf.TTSDataset import MyDataset 
from TTS_tf.layers.losses import loss_l1, loss_l2, stopnet_loss
from TTS_tf.utils.audio import AudioProcessor
from TTS_tf.utils.generic_utils import (check_gradient, count_parameters,
                                     create_experiment_folder, get_git_branch,
                                     load_config, remove_experiment_folder,
                                     copy_config_file, setup_model,
                                     split_dataset, gradual_training_scheduler, KeepAverage,
                                     save_checkpoint, save_best_model)
from TTS_tf.utils.logger import Logger
from TTS_tf.utils.speakers import load_speaker_mapping, save_speaker_mapping, \
    get_speakers
from TTS_tf.utils.synthesis import synthesis
from TTS_tf.utils.text.symbols import phonemes, symbols
from TTS_tf.utils.visual import plot_alignment, plot_spectrogram
from TTS_tf.preprocessors import load_meta_data
from TTS_tf.utils.measures import alignment_diagonal_score


np.random.seed(1)
tf.random.set_seed(1)
use_cuda = tf.test.is_gpu_available()
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


def setup_loader(ap, is_val=False, verbose=False):
    global meta_data_train
    global meta_data_eval
    if "meta_data_train" not in globals():
        meta_data_train, meta_data_eval = load_meta_data(c.datasets)
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            c.r,
            c.text_cleaner,
            meta_data=meta_data_eval[:64] if is_val else meta_data_train[:64],
            ap=ap,
            batch_group_size=0 if is_val else c.batch_group_size * c.batch_size,
            min_seq_len=c.min_seq_len,
            max_seq_len=c.max_seq_len,
            phoneme_cache_path=c.phoneme_cache_path,
            use_phonemes=c.use_phonemes,
            phoneme_language=c.phoneme_language,
            enable_eos_bos=c.enable_eos_bos_chars,
            verbose=verbose)
        # sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            num_workers=c.num_val_loader_workers
            if is_val else c.num_loader_workers,
            pin_memory=False)
    return loader


def train(model, criterion, criterion_st, optimizer, optimizer_st, scheduler,
          ap, global_step, epoch):
    data_loader = setup_loader(ap, is_val=False, verbose=(epoch == 0))
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)
    epoch_time = 0
    train_values = {
        'avg_postnet_loss': 0,
        'avg_decoder_loss': 0,
        'avg_stop_loss': 0,
        'avg_align_score': 0,
        'avg_step_time': 0,
        'avg_loader_time': 0,
        'avg_alignment_score': 0}
    keep_avg = KeepAverage()
    keep_avg.add_values(train_values)
    print("\n > Epoch {}/{}".format(epoch, c.epochs), flush=True)
    if use_cuda:
        batch_n_iter = int(len(data_loader.dataset) /
                           (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # setup input data
        text_input = data[0]
        text_lengths = data[1]
        speaker_names = data[2]
        linear_input = data[3] if c.model in [
            "Tacotron", "TacotronGST"] else None
        mel_input = data[4]
        mel_lengths = data[5]
        stop_targets = data[6]
        avg_text_length = np.mean(text_lengths)
        avg_spec_length = np.mean(mel_lengths)
        loader_time = time.time() - end_time

        if c.use_speaker_embedding:
            speaker_ids = [speaker_mapping[speaker_name]
                           for speaker_name in speaker_names]
        else:
            speaker_ids = None

        # set stop targets view, we predict a single stop token per r frames prediction
        stop_targets = stop_targets.reshape(text_input.shape[0],
                                         stop_targets.shape[1] // c.r, -1)
        stop_targets = tf.squeeze(stop_targets.sum(2) > 0.0)
        global_step += 1
        current_lr = optimizer.lr.numpy()

        with tf.GradientTape(persistent=True) as tape:
            # forward pass model
            decoder_output, postnet_output, alignments, stop_tokens = model(
                text_input, text_lengths, mel_input, speaker_ids=speaker_ids)
            
            # loss computation
            stop_loss = criterion_st(
                stop_tokens, stop_targets) if c.stopnet else 0.0
            if c.loss_masking:
                decoder_loss = criterion(
                    decoder_output, mel_input, mel_lengths)
                if c.model in ["Tacotron", "TacotronGST"]:
                    postnet_loss = criterion(
                        postnet_output, linear_input, mel_lengths)
                else:
                    postnet_loss = criterion(
                        postnet_output, mel_input, mel_lengths)
            else:
                decoder_loss = criterion(decoder_output, mel_input)
                if c.model in ["Tacotron", "TacotronGST"]:
                    postnet_loss = criterion(postnet_output, linear_input)
                else:
                    postnet_loss = criterion(postnet_output, mel_input)
            loss = decoder_loss + postnet_loss 
            
            if not c.separate_stopnet and c.stopnet:
                loss += stop_loss
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # compute alignment score
        align_score = alignment_diagonal_score(alignments.numpy())
        keep_avg.update_value('avg_align_score', align_score)

        # backpass and check the grad norm for stop loss
        if c.separate_stopnet:
            stopnet_grads = tape.gradient(stop_loss, model.decoder.stopnet.trainable_variables)
            optimizer_st.apply_gradients(zip(stopnet_grads, model.decoder.stopnet.trainable_variables))
        del tape

        step_time = time.time() - start_time
        epoch_time += step_time

        if global_step % c.print_step == 0:
            print(
                "   | > Step:{}/{}  GlobalStep:{}  PostnetLoss:{:.5f}  "
                "DecoderLoss:{:.5f}  StopLoss:{:.5f}  AlignScore:{:.4f}  "
                "AvgTextLen:{:.1f}  AvgSpecLen:{:.1f}  StepTime:{:.2f}  "
                "LoaderTime:{:.2f}  LR:{:.6f}".format(
                    num_iter, batch_n_iter, global_step,
                    postnet_loss, decoder_loss, stop_loss, align_score,
                    avg_text_length, avg_spec_length, step_time,
                    loader_time, current_lr),
                flush=True)

        # aggregate losses from processes
        # if num_gpus > 1:
        #     postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
        #     decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
        #     loss = reduce_tensor(loss.data, num_gpus)
        #     stop_loss = reduce_tensor(
        #         stop_loss.data, num_gpus) if c.stopnet else stop_loss
        if args.rank == 0:
            update_train_values = {'avg_postnet_loss': float(postnet_loss),
                                   'avg_decoder_loss': float(decoder_loss),
                                   'avg_stop_loss': float(stop_loss),
                                   'avg_step_time': step_time,
                                   'avg_loader_time': loader_time}
            keep_avg.update_values(update_train_values)

            # Plot Training Iter Stats
            # reduce TB load
            if global_step % 10 == 0:
                iter_stats = {"loss_posnet": float(postnet_loss),
                              "loss_decoder": float(decoder_loss),
                              "lr": float(current_lr),
                              "step_time": step_time}
                tb_logger.tb_train_iter_stats(global_step, iter_stats)

            if global_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    save_checkpoint(model, optimizer, optimizer_st,
                                    postnet_loss, OUT_PATH, global_step,
                                    epoch)

                # Diagnostic visualizations
                const_spec = postnet_output[0].numpy()
                gt_spec = linear_input[0] if c.model in [
                    "Tacotron", "TacotronGST"] else mel_input[0]
                align_img = alignments[0].numpy()

                figures = {
                    "prediction": plot_spectrogram(const_spec, ap),
                    "ground_truth": plot_spectrogram(gt_spec, ap),
                    "alignment": plot_alignment(align_img)
                }
                tb_logger.tb_train_figures(global_step, figures)

                # Sample audio
                if c.model in ["Tacotron", "TacotronGST"]:
                    train_audio = ap.inv_spectrogram(const_spec.T)
                else:
                    train_audio = ap.inv_mel_spectrogram(const_spec.T)
                tb_logger.tb_train_audios(global_step,
                                          {'TrainAudio': train_audio},
                                          c.audio["sample_rate"])
        end_time = time.time()
    # print epoch stats
    print(
        "   | > EPOCH END -- GlobalStep:{}  AvgTotalLoss:{:.5f}  "
        "AvgPostnetLoss:{:.5f}  AvgDecoderLoss:{:.5f}  "
        "AvgStopLoss:{:.5f}  EpochTime:{:.2f}  "
        "AvgStepTime:{:.2f}  AvgLoaderTime:{:.2f}".format(global_step, keep_avg['avg_postnet_loss'], keep_avg['avg_decoder_loss'],
                                                          keep_avg['avg_stop_loss'], keep_avg['avg_align_score'],
                                                          epoch_time, keep_avg['avg_step_time'], keep_avg['avg_loader_time']),
        flush=True)

    # Plot Epoch Stats
    if args.rank == 0:
        # Plot Training Epoch Stats
        epoch_stats = {"loss_postnet": keep_avg['avg_postnet_loss'],
                       "loss_decoder": keep_avg['avg_decoder_loss'],
                       "stop_loss": keep_avg['avg_stop_loss'],
                       "alignment_score": keep_avg['avg_align_score'],
                       "epoch_time": epoch_time}
        tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
        if c.tb_model_param_stats:
            tb_logger.tb_model_weights(model, global_step)
    return keep_avg['avg_postnet_loss'], global_step


def evaluate(model, criterion, criterion_st, ap, global_step, epoch):
    data_loader = setup_loader(ap, is_val=True)
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)
    epoch_time = 0
    eval_values_dict = {'avg_postnet_loss': 0,
                        'avg_decoder_loss': 0,
                        'avg_stop_loss': 0,
                        'avg_align_score': 0}
    keep_avg = KeepAverage()
    keep_avg.add_values(eval_values_dict)
    print("\n > Validation")
    if c.test_sentences_file is None:
        test_sentences = [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "Be a voice, not an echo.",
            "I'm sorry Dave. I'm afraid I can't do that.",
            "This cake is great. It's so delicious and moist."
        ]
    else:
        with open(c.test_sentences_file, "r") as f:
            test_sentences = [s.strip() for s in f.readlines()]
    if data_loader is not None:
        for num_iter, data in enumerate(data_loader):
            start_time = time.time()

            # setup input data
            text_input = data[0]
            text_lengths = data[1]
            speaker_names = data[2]
            linear_input = data[3] if c.model in [
                "Tacotron", "TacotronGST"] else None
            mel_input = data[4]
            mel_lengths = data[5]
            stop_targets = data[6]

            if c.use_speaker_embedding:
                speaker_ids = [speaker_mapping[speaker_name]
                                for speaker_name in speaker_names]
            else:
                speaker_ids = None

            # set stop targets view, we predict a single stop token per r frames prediction
            stop_targets = stop_targets.reshape(text_input.shape[0],
                                         stop_targets.shape[1] // c.r, -1)
            stop_targets = tf.squeeze(stop_targets.sum(2) > 0.0)

            # forward pass
            decoder_output, postnet_output, alignments, stop_tokens = model(
                text_input, text_lengths, mel_input, speaker_ids=speaker_ids)

            # loss computation
            stop_loss = criterion_st(
                stop_tokens, stop_targets) if c.stopnet else 0.0
            if c.loss_masking:
                decoder_loss = criterion(
                    decoder_output, mel_input, mel_lengths)
                if c.model in ["Tacotron", "TacotronGST"]:
                    postnet_loss = criterion(
                        postnet_output, linear_input, mel_lengths)
                else:
                    postnet_loss = criterion(
                        postnet_output, mel_input, mel_lengths)
            else:
                decoder_loss = criterion(decoder_output, mel_input)
                if c.model in ["Tacotron", "TacotronGST"]:
                    postnet_loss = criterion(postnet_output, linear_input)
                else:
                    postnet_loss = criterion(postnet_output, mel_input)
            loss = decoder_loss + postnet_loss 

            if not c.separate_stopnet and c.stopnet:
                loss += stop_loss
        
            step_time = time.time() - start_time
            epoch_time += step_time

            # compute alignment score
            align_score = alignment_diagonal_score(alignments.numpy())
            keep_avg.update_value('avg_align_score', align_score)

            # aggregate losses from processes
            if num_gpus > 1:
                postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
                decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
                if c.stopnet:
                    stop_loss = reduce_tensor(stop_loss.data, num_gpus)

            keep_avg.update_values({'avg_postnet_loss': float(postnet_loss),
                                    'avg_decoder_loss': float(decoder_loss),
                                    'avg_stop_loss': float(stop_loss)})

            if num_iter % c.print_step == 0:
                print(
                    "   | > TotalLoss: {:.5f}   PostnetLoss: {:.5f} - {:.5f}  DecoderLoss:{:.5f} - {:.5f} "
                    "StopLoss: {:.5f} - {:.5f}  AlignScore: {:.4f} : {:.4f}".format(
                        loss,
                        postnet_loss, keep_avg['avg_postnet_loss'],
                        decoder_loss, keep_avg['avg_decoder_loss'],
                        stop_loss, keep_avg['avg_stop_loss'],
                        align_score, keep_avg['avg_align_score']),
                    flush=True)

        if args.rank == 0:
            # Diagnostic visualizations
            idx = np.random.randint(mel_input.shape[0])
            const_spec = postnet_output[idx].numpy()
            gt_spec = linear_input[idx] if c.model in [
                "Tacotron", "TacotronGST"] else mel_input[idx]
            align_img = alignments[idx].numpy()

            eval_figures = {
                "prediction": plot_spectrogram(const_spec, ap),
                "ground_truth": plot_spectrogram(gt_spec, ap),
                "alignment": plot_alignment(align_img)
            }
            tb_logger.tb_eval_figures(global_step, eval_figures)

            # Sample audio
            if c.model in ["Tacotron", "TacotronGST"]:
                eval_audio = ap.inv_spectrogram(const_spec.T)
            else:
                eval_audio = ap.inv_mel_spectrogram(const_spec.T)
            tb_logger.tb_eval_audios(
                global_step, {"ValAudio": eval_audio}, c.audio["sample_rate"])

            # Plot Validation Stats
            epoch_stats = {"loss_postnet": keep_avg['avg_postnet_loss'],
                            "loss_decoder": keep_avg['avg_decoder_loss'],
                            "stop_loss": keep_avg['avg_stop_loss']}
            tb_logger.tb_eval_stats(global_step, epoch_stats)

    if args.rank == 0 and epoch >= c.test_delay_epochs:
        # test sentences
        test_audios = {}
        test_figures = {}
        print(" > Synthesizing test sentences")
        speaker_id = 0 if c.use_speaker_embedding else None
        style_wav = c.get("style_wav_for_test")
        for idx, test_sentence in enumerate(test_sentences):
            try:
                wav, alignment, decoder_output, postnet_output, stop_tokens = synthesis(
                    model, test_sentence, c, use_cuda, ap,
                    speaker_id=speaker_id,
                    style_wav=style_wav)
                file_path = os.path.join(AUDIO_PATH, str(global_step))
                os.makedirs(file_path, exist_ok=True)
                file_path = os.path.join(file_path,
                                         "TestSentence_{}.wav".format(idx))
                ap.save_wav(wav, file_path)
                test_audios['{}-audio'.format(idx)] = wav
                test_figures['{}-prediction'.format(idx)
                             ] = plot_spectrogram(postnet_output, ap)
                test_figures['{}-alignment'.format(idx)
                             ] = plot_alignment(alignment)
            except:
                print(" !! Error creating Test Sentence -", idx)
                traceback.print_exc()
            tb_logger.tb_test_audios(
                global_step, test_audios, c.audio['sample_rate'])
        tb_logger.tb_test_figures(global_step, test_figures)
    return keep_avg['avg_postnet_loss']


# FIXME: move args definition/parsing inside of main?
def main(args):  # pylint: disable=redefined-outer-name
    # Audio processor
    ap = AudioProcessor(**c.audio)

    # DISTRUBUTED
    if num_gpus > 1:
        #TODO: multi-GPU
        pass
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    if c.use_speaker_embedding:
        speakers = get_speakers(c.data_path, c.meta_file_train, c.dataset)
        if args.restore_path:
            prev_out_path = os.path.dirname(args.restore_path)
            speaker_mapping = load_speaker_mapping(prev_out_path)
            assert all([speaker in speaker_mapping
                        for speaker in speakers]), "As of now you, you cannot " \
                                                   "introduce new speakers to " \
                                                   "a previously trained model."
        else:
            speaker_mapping = {name: i
                               for i, name in enumerate(speakers)}
        save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print("Training with {} speakers: {}".format(num_speakers,
                                                     ", ".join(speakers)))
    else:
        num_speakers = 0

    model = setup_model(num_chars, num_speakers, c)

    print(" | > Num output units : {}".format(ap.num_freq), flush=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=c.lr, clipnorm=c.grad_clip)

    if c.stopnet and c.separate_stopnet:
        optimizer_st = tf.keras.optimizers.Adam(learning_rate=0.001)
    else:
        optimizer_st = None

    criterion = loss_l1 if c.model in [
        "Tacotron", "TacotronGST"] else l2_loss
    
    criterion_st = stopnet_loss if c.stopnet else None

    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        for group in optimizer.param_groups:
            group['lr'] = c.lr
        print(
            " > Model restored from step %d" % checkpoint['step'], flush=True)
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    # DISTRUBUTED
    # if num_gpus > 1:
    #     model = apply_gradient_allreduce(model)

    # if c.lr_decay:
    #     scheduler = NoamLR(
    #         optimizer,
    #         warmup_steps=c.warmup_steps,
    #         last_epoch=args.restore_step - 1)
    # else:
    # TODO: scheduler 
    scheduler = None

    num_params = count_parameters(model, c)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    global_step = args.restore_step
    for epoch in range(0, c.epochs):
        # set gradual training
        if c.gradual_training is not None:
            r, c.batch_size = gradual_training_scheduler(global_step, c)
            c.r = r
            model.decoder.set_r(r)
        print(" > Number of outputs per iteration:", model.decoder.r)

        train_loss, global_step = train(model, criterion, criterion_st,
                                        optimizer, optimizer_st, scheduler,
                                        ap, global_step, epoch)
        val_loss = evaluate(model, criterion, criterion_st,
                            ap, global_step, epoch)
        print(
            " | > Training Loss: {:.5f}   Validation Loss: {:.5f}".format(
                train_loss, val_loss),
            flush=True)
        target_loss = train_loss
        if c.run_eval:
            target_loss = val_loss
        best_loss = save_best_model(model, optimizer, target_loss, best_loss,
                                    OUT_PATH, global_step, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Path to model outputs (checkpoint, tensorboard etc.).',
        default=0)
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=True,
        help='Do not verify commit integrity to run training.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='Defines the data path. It overwrites config.json.')
    parser.add_argument(
        '--output_path',
        type=str,
        help='path for training outputs.',
        default='')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='',
        help='folder name for training outputs.'
    )

    # DISTRUBUTED
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='DISTRIBUTED: process rank for distributed training.')
    parser.add_argument(
        '--group_id',
        type=str,
        default="",
        help='DISTRIBUTED: process group id.')
    args = parser.parse_args()

    # setup output paths and read configs
    c = load_config(args.config_path)
    _ = os.path.dirname(os.path.realpath(__file__))
    if args.data_path != '':
        c.data_path = args.data_path

    if args.output_path == '':
        OUT_PATH = os.path.join(_, c.output_path)
    else:
        OUT_PATH = args.output_path

    if args.group_id == '' and args.output_folder == '':
        OUT_PATH = create_experiment_folder(OUT_PATH, c.run_name, args.debug)
    else:
        OUT_PATH = os.path.join(OUT_PATH, args.output_folder)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_config_file(args.config_path, os.path.join(
            OUT_PATH, 'config.json'), new_fields)
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

    if args.rank == 0:
        LOG_DIR = OUT_PATH
        tb_logger = Logger(LOG_DIR)

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
