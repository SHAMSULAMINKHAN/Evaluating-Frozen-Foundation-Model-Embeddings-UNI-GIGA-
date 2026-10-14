from pathlib import Path
import csv
import json
import gc

import numpy as np
import faiss


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing WSI .npy feature files
INPUT_FOLDER = Path(r"D:\datasets\LUAD")

# Folder where clustering results will be saved
OUTPUT_FOLDER = Path(r"D:\datasets\output\kmeans")

# Number of clusters
K = 5

# Number of FAISS KMeans iterations
NITER = 50

# Use FAISS GPU?
USE_GPU = False

# ------------------------------------------------------------
# MEMORY SETTINGS
# ------------------------------------------------------------

# Number of embeddings processed at one time during
# normalization / assignment.
BATCH_SIZE = 16384

# Number of embeddings used to TRAIN KMeans.
#
# IMPORTANT:
# FAISS Kmeans.train() normally expects all training vectors
# in memory. To avoid loading all 3+ million embeddings,
# we train on a representative random sample instead.
#
# After training, ALL embeddings are assigned to clusters
# in batches.
TRAIN_SAMPLE_SIZE = 3_080_719

# Random seed for reproducible sampling
RANDOM_SEED = 42

# ------------------------------------------------------------
# OUTPUT SETTINGS
# ------------------------------------------------------------

# Save a CSV containing:
# embedding_name, WSI, patch_id, patch_name, cluster, distance
SAVE_MAPPING_CSV = True

# Save labels/distances as NumPy .npy files
SAVE_LABELS = True
SAVE_DISTANCES = True


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("KIRC EMBEDDING CLUSTERING - MEMORY EFFICIENT VERSION")
print("=" * 70)

print(f"\nInput folder       : {INPUT_FOLDER}")
print(f"Output folder      : {OUTPUT_FOLDER}")
print(f"Number of clusters : {K}")
print(f"KMeans iterations  : {NITER}")
print(f"FAISS GPU          : {USE_GPU}")
print(f"Batch size         : {BATCH_SIZE:,}")
print(f"Training sample    : {TRAIN_SAMPLE_SIZE:,}")
print(f"Random seed        : {RANDOM_SEED}")


# ============================================================
# CHECK INPUT FOLDER
# ============================================================

if not INPUT_FOLDER.exists():
    raise FileNotFoundError(
        f"\nInput folder does not exist:\n{INPUT_FOLDER}"
    )


# ============================================================
# FIND ALL .NPY FILES
# ============================================================

wsi_files = sorted(
    INPUT_FOLDER.glob("*.npy")
)

if len(wsi_files) == 0:
    raise FileNotFoundError(
        f"\nNo .npy files found in:\n{INPUT_FOLDER}"
    )

print(
    f"\nFound {len(wsi_files)} .npy files."
)


# ============================================================
# FIRST PASS
#
# Read each WSI only to determine:
#   - number of patches
#   - embedding dimension
#   - valid files
#
# We DO NOT keep embeddings in RAM.
# ============================================================

print("\n" + "=" * 70)
print("FIRST PASS - CHECKING WSI FILES")
print("=" * 70)

file_info = []
failed_files = []

total_patches = 0
embedding_dim = None


for wsi_file in wsi_files:

    wsi_name = wsi_file.stem

    print(f"\nChecking: {wsi_name}")

    try:

        data = np.load(
            wsi_file,
            allow_pickle=True
        ).item()

        if not isinstance(data, dict):
            raise ValueError(
                "File does not contain a dictionary."
            )

        if "features" not in data:
            raise ValueError(
                "Missing 'features' key."
            )

        if "patch_names" not in data:
            raise ValueError(
                "Missing 'patch_names' key."
            )

        embeddings = np.asarray(
            data["features"]
        )

        patch_names = data["patch_names"]

        if embeddings.ndim != 2:
            raise ValueError(
                f"Invalid embedding shape: {embeddings.shape}"
            )

        if len(embeddings) != len(patch_names):
            raise ValueError(
                f"Embedding/patch mismatch: "
                f"{len(embeddings)} embeddings vs "
                f"{len(patch_names)} patch names"
            )

        if len(embeddings) == 0:
            raise ValueError(
                "No embeddings found in file."
            )

        current_dim = embeddings.shape[1]

        if embedding_dim is None:
            embedding_dim = current_dim

        elif current_dim != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {embedding_dim}, got {current_dim}"
            )

        # Check values without creating a float32 copy
        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain NaN or Inf values."
            )

        n_patches = len(embeddings)

        file_info.append({
            "path": str(wsi_file),
            "wsi": wsi_name,
            "num_patches": int(n_patches),
        })

        total_patches += n_patches

        print(
            f"  OK - {n_patches:,} patches, "
            f"dimension={current_dim}"
        )

        del data
        del embeddings
        del patch_names
        gc.collect()

    except Exception as e:

        print("  FAILED")
        print(
            f"  Error: {type(e).__name__}: {e}"
        )

        failed_files.append({
            "file": str(wsi_file),
            "wsi": wsi_name,
            "error_type": type(e).__name__,
            "error": str(e)
        })


# ============================================================
# FIRST PASS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FIRST PASS SUMMARY")
print("=" * 70)

print(
    f"Total WSI files     : {len(wsi_files)}"
)

print(
    f"Successfully loaded : {len(file_info)}"
)

print(
    f"Failed files        : {len(failed_files)}"
)

print(
    f"Valid patches       : {total_patches:,}"
)

print(
    f"Embedding dimension : {embedding_dim}"
)


if len(file_info) == 0:
    raise RuntimeError(
        "\nNo valid WSI files were loaded."
    )


# ============================================================
# SAVE FAILED FILE LIST
# ============================================================

if failed_files:

    failed_path = (
        OUTPUT_FOLDER / "failed_files.json"
    )

    with open(
        failed_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            failed_files,
            f,
            indent=4
        )

    print(
        f"\nFailed file list saved to:"
    )

    print(
        f"  {failed_path}"
    )


# ============================================================
# CREATE TRAINING SAMPLE
#
# We use random global patch positions so that the sample
# represents the whole dataset rather than only the first WSI.
# ============================================================

print("\n" + "=" * 70)
print("CREATING KMEANS TRAINING SAMPLE")
print("=" * 70)

rng = np.random.default_rng(RANDOM_SEED)

sample_size = min(
    TRAIN_SAMPLE_SIZE,
    total_patches
)

# Random global positions
sample_positions = np.sort(
    rng.choice(
        total_patches,
        size=sample_size,
        replace=False
    )
)

print(
    f"Training sample size: {sample_size:,}"
)


# ============================================================
# LOAD TRAINING SAMPLE
#
# Only TRAIN_SAMPLE_SIZE embeddings are kept in RAM.
# ============================================================

train_X = np.empty(
    (sample_size, embedding_dim),
    dtype=np.float32
)

sample_cursor = 0
global_offset = 0


for info in file_info:

    if sample_cursor >= sample_size:
        break

    start_global = global_offset
    end_global = (
        global_offset + info["num_patches"]
    )

    # Find sampled positions belonging to this WSI
    left = np.searchsorted(
        sample_positions,
        start_global,
        side="left"
    )

    right = np.searchsorted(
        sample_positions,
        end_global,
        side="left"
    )

    if right > left:

        local_positions = (
            sample_positions[left:right]
            - start_global
        )

        data = np.load(
            info["path"],
            allow_pickle=True
        ).item()

        embeddings = np.asarray(
            data["features"],
            dtype=np.float32
        )

        selected = embeddings[
            local_positions
        ]

        n_selected = len(selected)

        train_X[
            sample_cursor:
            sample_cursor + n_selected
        ] = selected

        sample_cursor += n_selected

        del data
        del embeddings
        del selected
        gc.collect()

    global_offset = end_global

    print(
        f"  Sample loaded: "
        f"{sample_cursor:,}/{sample_size:,}"
    )


# ============================================================
# NORMALIZE TRAINING SAMPLE
# ============================================================

print("\nNormalizing training sample...")

# L2 normalization in-place.
norms = np.linalg.norm(
    train_X,
    axis=1,
    keepdims=True
)

norms[norms == 0] = 1.0

train_X /= norms

del norms
gc.collect()


# ============================================================
# TRAIN FAISS KMEANS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FAISS KMEANS")
print("=" * 70)

d = embedding_dim

print(
    f"Dimension        : {d}"
)

print(
    f"Clusters         : {K}"
)

print(
    f"Iterations       : {NITER}"
)

print(
    f"Training vectors : {len(train_X):,}"
)

print(
    f"GPU              : {USE_GPU}"
)


if K <= 1:
    raise ValueError(
        "K must be greater than 1."
    )

if K > len(train_X):
    raise ValueError(
        f"K ({K}) cannot be greater than "
        f"training sample size ({len(train_X)})."
    )


kmeans = faiss.Kmeans(
    d=d,
    k=K,
    niter=NITER,
    verbose=True,
    gpu=USE_GPU,
    seed=RANDOM_SEED
)

print("\nTraining KMeans...")

kmeans.train(train_X)

print(
    "\nKMeans training completed."
)


# Training sample is no longer needed.
del train_X
del sample_positions
gc.collect()


# ============================================================
# CLUSTER CENTERS
# ============================================================

centers = np.asarray(
    kmeans.centroids,
    dtype=np.float32
)

print(
    f"\nCluster centers shape: {centers.shape}"
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

print("\n" + "=" * 70)
print("SAVING KMEANS")
print("=" * 70)

faiss_index_path = (
    OUTPUT_FOLDER / "kmeans_faiss.index"
)

faiss.write_index(
    kmeans.index,
    str(faiss_index_path)
)

print(
    f"\nSaved FAISS index:"
)

print(
    f"  {faiss_index_path}"
)


# ============================================================
# SAVE CLUSTER CENTERS
# ============================================================

centers_path = (
    OUTPUT_FOLDER / "kmeans_centers.npy"
)

np.save(
    centers_path,
    centers
)

print(
    f"\nSaved cluster centers:"
)

print(
    f"  {centers_path}"
)


# ============================================================
# PREPARE OUTPUT ARRAYS
#
# These are stored as .npy memory-mapped arrays, so labels
# and distances do not have to consume large amounts of RAM.
# ============================================================

labels_path = (
    OUTPUT_FOLDER / "labels.npy"
)

distances_path = (
    OUTPUT_FOLDER / "distances.npy"
)


labels = None
distances = None

if SAVE_LABELS:

    labels = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_patches,)
    )

if SAVE_DISTANCES:

    distances = np.lib.format.open_memmap(
        distances_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_patches,)
    )


# ============================================================
# MAPPING CSV
#
# This replaces the huge in-memory embedding_names and
# patch_metadata lists from the original script.
# ============================================================

mapping_path = (
    OUTPUT_FOLDER /
    "embedding_cluster_mapping.csv"
)

mapping_file = None
mapping_writer = None

if SAVE_MAPPING_CSV:

    mapping_file = open(
        mapping_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    mapping_writer = csv.writer(
        mapping_file
    )

    mapping_writer.writerow([
        "global_index",
        "embedding_name",
        "wsi",
        "patch",
        "patch_name",
        "cluster",
        "distance"
    ])


# ============================================================
# ASSIGN ALL EMBEDDINGS
#
# This is the important memory-saving part.
#
# We NEVER create:
#
#     all_embeddings = np.concatenate(...)
#
# and we NEVER create:
#
#     X = normalize(all_embeddings)
#
# Instead, each WSI is processed in BATCH_SIZE chunks.
# ============================================================

print("\n" + "=" * 70)
print("ASSIGNING ALL EMBEDDINGS TO CLUSTERS")
print("=" * 70)

global_index = 0


for file_number, info in enumerate(file_info, start=1):

    print(
        f"\n[{file_number}/{len(file_info)}] "
        f"Processing: {info['wsi']}"
    )

    data = np.load(
        info["path"],
        allow_pickle=True
    ).item()

    embeddings = np.asarray(
        data["features"],
        dtype=np.float32
    )

    patch_names = data["patch_names"]

    n_embeddings = len(embeddings)

    print(
        f"  Embeddings: {n_embeddings:,}"
    )

    for start in range(
        0,
        n_embeddings,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            n_embeddings
        )

        # Copy only this batch.
        batch = np.array(
            embeddings[start:end],
            dtype=np.float32,
            copy=True
        )

        # ----------------------------------------------------
        # L2 NORMALIZE BATCH IN-PLACE
        # ----------------------------------------------------

        norms = np.linalg.norm(
            batch,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1.0

        batch /= norms

        del norms

        # ----------------------------------------------------
        # ASSIGN BATCH TO NEAREST CLUSTER
        # ----------------------------------------------------

        batch_distances, batch_labels = (
            kmeans.index.search(
                batch,
                1
            )
        )

        batch_labels = (
            batch_labels.reshape(-1)
        )

        batch_distances = (
            batch_distances.reshape(-1)
            .astype(np.float32)
        )

        batch_global_start = (
            global_index + start
        )

        batch_global_end = (
            global_index + end
        )

        # ----------------------------------------------------
        # SAVE LABELS / DISTANCES
        # ----------------------------------------------------

        if labels is not None:

            labels[
                batch_global_start:
                batch_global_end
            ] = batch_labels

        if distances is not None:

            distances[
                batch_global_start:
                batch_global_end
            ] = batch_distances

        # ----------------------------------------------------
        # SAVE MAPPING DIRECTLY TO CSV
        # ----------------------------------------------------

        if mapping_writer is not None:

            for local_index in range(
                end - start
            ):

                patch_index = (
                    start + local_index
                )

                patch_name = str(
                    patch_names[patch_index]
                )

                mapping_writer.writerow([
                    global_index + patch_index,
                    patch_name,
                    info["wsi"],
                    patch_index,
                    patch_name,
                    int(batch_labels[local_index]),
                    float(batch_distances[local_index])
                ])

        # ----------------------------------------------------
        # CLEAN BATCH
        # ----------------------------------------------------

        del batch
        del batch_distances
        del batch_labels

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        print(
            f"    {end:,}/{n_embeddings:,} "
            f"processed",
            end="\r"
        )

    print()

    global_index += n_embeddings

    del data
    del embeddings
    del patch_names

    gc.collect()


# ============================================================
# FLUSH OUTPUTS
# ============================================================

if labels is not None:
    labels.flush()

if distances is not None:
    distances.flush()

if mapping_file is not None:
    mapping_file.close()


# ============================================================
# CLUSTER COUNTS
#
# Count labels without creating a giant temporary array.
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER COUNTS")
print("=" * 70)

cluster_counts = np.zeros(
    K,
    dtype=np.int64
)

if labels is not None:

    for start in range(
        0,
        total_patches,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total_patches
        )

        batch_labels = labels[
            start:end
        ]

        cluster_counts += np.bincount(
            batch_labels,
            minlength=K
        )

else:
    raise RuntimeError(
        "Labels were not saved, so cluster counts "
        "cannot be calculated."
    )


for cluster_id in range(K):

    print(
        f"  Cluster {cluster_id}: "
        f"{cluster_counts[cluster_id]:,} embeddings"
    )


# ============================================================
# CONFIGURATION
# ============================================================

config = {

    "input_folder": str(
        INPUT_FOLDER
    ),

    "output_folder": str(
        OUTPUT_FOLDER
    ),

    "embedding_dimension": int(
        embedding_dim
    ),

    "number_of_embeddings": int(
        total_patches
    ),

    "number_of_clusters": int(
        K
    ),

    "kmeans_iterations": int(
        NITER
    ),

    "batch_size": int(
        BATCH_SIZE
    ),

    "training_sample_size": int(
        sample_size
    ),

    "random_seed": int(
        RANDOM_SEED
    ),

    "normalized": True,

    "normalization": "L2",

    "faiss_gpu_training": bool(
        USE_GPU
    ),

    "num_wsi_files_found": int(
        len(wsi_files)
    ),

    "num_wsi_files_loaded": int(
        len(file_info)
    ),

    "num_wsi_files_failed": int(
        len(failed_files)
    ),

    "cluster_counts": (
        cluster_counts.tolist()
    ),

    "mapping_format": (
        "CSV"
        if SAVE_MAPPING_CSV
        else None
    )

}


config_path = (
    OUTPUT_FOLDER /
    "clustering_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLUSTERING COMPLETED")
print("=" * 70)

print(
    f"\nWSI files found     : {len(wsi_files)}"
)

print(
    f"WSI files loaded    : {len(file_info)}"
)

print(
    f"WSI files failed    : {len(failed_files)}"
)

print(
    f"Total embeddings    : {total_patches:,}"
)

print(
    f"Embedding dimension : {embedding_dim}"
)

print(
    f"Number of clusters  : {K}"
)

print(
    "\nCluster distribution:"
)

for cluster_id in range(K):

    percentage = (
        cluster_counts[cluster_id]
        / total_patches
        * 100
    )

    print(
        f"  Cluster {cluster_id}: "
        f"{cluster_counts[cluster_id]:,} "
        f"({percentage:.2f}%)"
    )


print("\nSaved files:")

print(
    f"  1. {faiss_index_path.name}"
)

print(
    f"  2. {centers_path.name}"
)

if SAVE_LABELS:
    print(
        f"  3. {labels_path.name}"
    )

if SAVE_DISTANCES:
    print(
        f"  4. {distances_path.name}"
    )

if SAVE_MAPPING_CSV:
    print(
        f"  5. {mapping_path.name}"
    )

print(
    f"  6. {config_path.name}"
)

if failed_files:

    print(
        f"  7. failed_files.json"
    )

print(
    f"\nOutput directory:"
)

print(
    f"  {OUTPUT_FOLDER}"
)

print("\nDone.")
