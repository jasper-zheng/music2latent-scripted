from .fad import FrechetAudioDistance
from .hparams import hparams

def compute_fad(fad_path):
    scores = []
    for i,model_name in enumerate(hparams.fad_models):

        if model_name == 'vggish':
            frechet = FrechetAudioDistance(
                model_name="vggish",
                sample_rate=16000,
                use_pca=False, 
                use_activation=False,
                verbose=True,
                audio_load_worker=hparams.fad_workers)
        elif model_name == 'clap':
            frechet = FrechetAudioDistance(
                model_name="clap",
                sample_rate=48000,
                submodel_name="630k-audioset",
                verbose=True,
                enable_fusion=False,
                audio_load_worker=hparams.fad_workers)
        else:
            raise NameError('Must be (vggish, clap)')

        score = frechet.score(
            hparams.data_path_test,
            fad_path,
            background_embds_path=hparams.fad_background_embeddings[i],
            dtype="float32")
        scores.append(score)
        
    return scores