import numpy as np
import pandas as pd
import polars as pl
from soundsig.sound import BioSound, WavFile
from multiprocessing import Pool, cpu_count
from functools import partial
import re
import os
import random
from tqdm import tqdm

def analyze_distribution_shape_right(data, bins, thresholds=(0.5, 0.25, 0.1)):
    """Identifies the peak of the data distribution and finds x-values for right-tail thresholds."""
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peak_index = np.argmax(counts)
    peak_x = bin_centers[peak_index]
    peak_count = counts[peak_index]
    
    threshold_counts = [peak_count * t for t in thresholds]
    x_values = {}
    for i, threshold_count in enumerate(threshold_counts):
        right_index = np.where(counts[peak_index:] <= threshold_count)[0]
        if len(right_index) > 0:
            right_x = bin_centers[peak_index + right_index[0]]
        else:
            right_x = bin_edges[-1]
        x_values[thresholds[i]] = right_x
    
    return {'peak': {'x': peak_x, 'count': peak_count}, 'thresholds': x_values}

def process_sound_file(fname, meta_data, window_size=0.2, normalize=False):
    """Loads a WAV file, pads/normalizes it, computes MPS features, and retrieves metadata."""
    try:
        filename, _ = os.path.splitext(fname)
        parts = filename.split('_')
        segment = parts[-1]
        id_ = parts[-2]

        soundIn = WavFile(file_name=fname, mono=True)
        data = soundIn.data.astype(np.float32)
        sample_rate = float(soundIn.sample_rate)
        del soundIn.data

        # Pad signal to ensure it meets the minimum window size
        max_window_size = window_size * 1.01
        max_window_samples = int(max_window_size * sample_rate)
        if len(data) < max_window_samples:
            pad_size = (max_window_samples - len(data)) // 2
            pad_right = pad_size + (max_window_samples - len(data)) % 2
            data = np.pad(data, (pad_size, pad_right), mode='constant')
        
        if normalize:
            maxAmp = np.abs(data).max()
        else:
            maxAmp = 1.0

        myBioSound = BioSound(
            soundWave=data / maxAmp,
            fs=sample_rate,
            emitter="",
            calltype=parts[-1]
        )

        del data

        # Compute Modulation Power Spectrum (MPS)
        myBioSound.mpsCalc(window=window_size, Norm=True)

        wfInd = np.argwhere(myBioSound.wf >= 0).flatten()
        wtInd = np.argwhere((myBioSound.wt >= -100) & (myBioSound.wt <= 100)).flatten()

        wfi, wtj = np.meshgrid(wfInd, wtInd, indexing='ij')
        mps_data = np.log(np.ravel(myBioSound.mps[wfi, wtj]))

        # Retrieve associated metadata using Polars
        matched_rows = meta_data.filter(pl.col("id") == int(id_))
        gen, family, species_meta, sub_species, common_name = (None,)*5
        recordist, date, time, country, location, lat, lng, bird = (None,)*8

        if not matched_rows.is_empty():
            row = matched_rows.row(0, named=True)
            gen, family = row['gen'], row['family']
            species_meta, sub_species = row['species'], row['ssp']
            common_name, recordist = row['common_name'], row['recordist']
            date, time = row['date'], row['time']
            country, location = row['country'], row['location']
            lat, lng = row['lat'], row['lng']
            bird = row['id']

        ex = (
            myBioSound.wt[wtInd[0]],
            myBioSound.wt[wtInd[-1]],
            myBioSound.wf[wfInd[0]] * 1e3,
            myBioSound.wf[wfInd[-1]] * 1e3
        )

        del myBioSound
        return (
            mps_data, gen, family, species_meta, sub_species, common_name, 
            recordist, date, time, country, location, lat, lng, bird,
            fname, ex, wfInd, wtInd
        )
    
    except Exception as e:
        print(f"Error processing {fname}: {str(e)}")
        return None

def main(input_dir, meta_data_filepath, normalize=True, downsample=False, window_size=0.2):
    os.chdir(input_dir)
    wav_files = [fname for fname in os.listdir('.') if fname.endswith('.wav')]
    
    if not downsample:
        allowed_files = wav_files
        meta_data_pl = pl.read_csv(meta_data_filepath)
        filtered_meta_pl = meta_data_pl
    else:
        # Adaptive Downsampling: Balance recordings per species based on file count distribution
        file_species_recording = {}
        for fname in wav_files:
            match = re.match(r"([A-Za-z]+_[A-Za-z]+)_(\d+)_seg(\d+)\.wav", fname)
            if match:
                sp, rec, seg = match.groups()
                file_species_recording.setdefault(sp, {}).setdefault(rec, []).append(fname)

        species_file_counts = {
            sp: sum(len(rec_list) for rec_list in recordings.values())
            for sp, recordings in file_species_recording.items()
        }

        # Calculate dynamic threshold based on distribution tail
        results = analyze_distribution_shape_right(list(species_file_counts.values()), bins=100, thresholds=[0.10])
        threshold_species = int(results['thresholds'][0.10])
        
        files_per_recording = {}
        allowed_files = []
        
        # Allocate quotas per recording to meet species threshold
        for sp, rec_dict in file_species_recording.items():
            species_total = species_file_counts[sp]
            num_recordings = len(rec_dict)
            files_per_recording[sp] = {}

            if species_total <= threshold_species:
                for rec in rec_dict:
                    files_per_recording[sp][rec] = len(rec_dict[rec])
            elif num_recordings <= threshold_species:
                recordings = list(rec_dict.keys())
                base_per_recording = threshold_species // len(recordings)
                recordings_with_excess = []
                
                for rec in recordings:
                    available = len(rec_dict[rec])
                    if available >= base_per_recording:
                        files_per_recording[sp][rec] = base_per_recording
                        if available > base_per_recording:
                            recordings_with_excess.append(rec)
                    else:
                        files_per_recording[sp][rec] = available
                
                total_allocated = sum(files_per_recording[sp][rec] for rec in recordings)
                remaining = threshold_species - total_allocated
                
                if remaining > 0 and recordings_with_excess:
                    extra_per_recording = remaining // len(recordings_with_excess)
                    final_extra = remaining % len(recordings_with_excess)
                    
                    for rec in recordings_with_excess:
                        files_per_recording[sp][rec] += extra_per_recording
                    
                    if final_extra:
                        lucky_recording = random.choice(recordings_with_excess)
                        files_per_recording[sp][lucky_recording] += final_extra
            else:
                chosen_recordings = random.sample(list(rec_dict.keys()), threshold_species)
                for rec in rec_dict:
                    files_per_recording[sp][rec] = 0
                for rec in chosen_recordings:
                    files_per_recording[sp][rec] = 1

        for sp, rec_dict in file_species_recording.items():
            for rec, file_list in rec_dict.items():
                quota = files_per_recording[sp][rec]
                if quota >= len(file_list):
                    allowed_files.extend(file_list)
                else:
                    allowed_files.extend(random.sample(file_list, quota))

        allowed_ids = {int(os.path.splitext(f)[0].split('_')[-2]) for f in allowed_files}
        meta_data_pl = pl.read_csv(meta_data_filepath)
        filtered_meta_pl = meta_data_pl.filter(pl.col("id").is_in(list(allowed_ids)))

    process_func = partial(
        process_sound_file,
        meta_data=filtered_meta_pl,
        window_size=window_size,
        normalize=normalize
    )

    num_processes = cpu_count()
    print(f"\nUsing {num_processes} processes")
    
    results_lists = {
        'mps_data': [], 'gen': [], 'family': [], 'species': [], 
        'sub_species': [], 'common_name': [], 'recordist': [], 
        'date': [], 'time': [], 'country': [], 'location': [], 
        'lat': [], 'lng': [], 'bird': [], 'filename': []
    }
    first_ex = first_wfInd = first_wtInd = None

    # Parallel processing of all allowed files
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(allowed_files), desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_func, allowed_files, chunksize=10):
                if result is not None:
                    (mps_data, gen, family, species, sub_species, common_name,
                     recordist, date, time, country, location, lat, lng, bird,
                     file_val, ex, wfInd, wtInd) = result

                    for key, value in zip(results_lists.keys(), 
                        [mps_data, gen, family, species, sub_species, common_name,
                         recordist, date, time, country, location, lat, lng, bird, file_val]):
                        results_lists[key].append(value)

                    if first_ex is None:
                        first_ex, first_wfInd, first_wtInd = ex, wfInd, wtInd

                pbar.update(1)

    # Convert features to numpy array and save
    X = np.array(results_lists['mps_data'], dtype=object)
    del results_lists['mps_data']

    np.save('X.npy', X)
    np.savez('metadata.npz', **results_lists)

    if first_ex is not None:
        np.savez('ex_indices.npz', ex=first_ex, wfInd=first_wfInd, wtInd=first_wtInd)
    else:
        print("Warning: Could not save 'ex', 'wfInd', 'wtInd' (no valid BioSound object)")

    print("\nProcessing complete!")

if __name__ == "__main__":
    input_dir = '../data/segments_passerines'
    meta_data_filepath = '../data/metadata_xeno_passerines.csv'
    main(input_dir, meta_data_filepath, normalize=True, downsample=False, window_size=0.1)
