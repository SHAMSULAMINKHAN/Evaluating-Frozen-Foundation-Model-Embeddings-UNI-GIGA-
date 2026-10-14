from pathlib import Path
import json

import numpy as np
import faiss
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing WSI .npy feature files
INPUT_FOLDER = Path(r"D:\datasets\features_LUSC")

# Folder where clustering results will be saved
OUTPUT_FOLDER = Path(r"D:\datasets\output\kmeans")

# Number of clusters
K = 5

# Number of KMeans iterations
NITER = 50

# Use FAISS GPU?
# False = CPU
# True = GPU
USE_GPU = False


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
print("KIRP EMBEDDING CLUSTERING")
print("=" * 70)

print(f"\nInput folder : {INPUT_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Number of clusters: {K}")
print(f"KMeans iterations : {NITER}")
print(f"FAISS GPU         : {USE_GPU}")


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
# LOAD ALL EMBEDDINGS
# ============================================================

all_embeddings = []

embedding_names = []

patch_metadata = []

failed_files = []

total_patches = 0

successful_files = 0


print("\n" + "=" * 70)
print("LOADING WSI EMBEDDINGS")
print("=" * 70)


for wsi_file in wsi_files:

    # --------------------------------------------------------
    # WSI name
    # --------------------------------------------------------

    wsi_name = wsi_file.stem

    print(f"\nLoading: {wsi_name}")

    try:

        # ----------------------------------------------------
        # Load file
        # ----------------------------------------------------

        data = np.load(
            wsi_file,
            allow_pickle=True
        ).item()


        # ----------------------------------------------------
        # Check dictionary
        # ----------------------------------------------------

        if not isinstance(data, dict):

            raise ValueError(
                "File does not contain a dictionary."
            )


        # ----------------------------------------------------
        # Check required fields
        # ----------------------------------------------------

        if "features" not in data:

            raise ValueError(
                "Missing 'features' key."
            )


        if "patch_names" not in data:

            raise ValueError(
                "Missing 'patch_names' key."
            )


        # ----------------------------------------------------
        # Extract data
        # ----------------------------------------------------

        embeddings = data["features"]

        patch_names = data["patch_names"]


        # ----------------------------------------------------
        # Convert embeddings
        # ----------------------------------------------------

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Validate embedding shape
        # ----------------------------------------------------

        if embeddings.ndim != 2:

            raise ValueError(
                f"Invalid embedding shape: "
                f"{embeddings.shape}"
            )


        # ----------------------------------------------------
        # Validate patch names
        # ----------------------------------------------------

        if len(embeddings) != len(patch_names):

            raise ValueError(
                f"Embedding/patch mismatch: "
                f"{len(embeddings)} embeddings vs "
                f"{len(patch_names)} patch names"
            )


        # ----------------------------------------------------
        # Check for empty file/data
        # ----------------------------------------------------

        if len(embeddings) == 0:

            raise ValueError(
                "No embeddings found in file."
            )


        # ----------------------------------------------------
        # Check for NaN / Inf
        # ----------------------------------------------------

        if not np.isfinite(embeddings).all():

            raise ValueError(
                "Embeddings contain NaN or Inf values."
            )


        # ----------------------------------------------------
        # Store embeddings
        # ----------------------------------------------------

        all_embeddings.append(
            embeddings
        )


        # ----------------------------------------------------
        # Store metadata
        # ----------------------------------------------------

        for patch_id, patch_name in enumerate(
            patch_names
        ):

            embedding_names.append(
                str(patch_name)
            )

            patch_metadata.append({

                "wsi": wsi_name,

                "patch": int(patch_id),

                "patch_name": str(patch_name)

            })


        # ----------------------------------------------------
        # Update counters
        # ----------------------------------------------------

        total_patches += len(embeddings)

        successful_files += 1


        # ----------------------------------------------------
        # Print success
        # ----------------------------------------------------

        print(
            f"  OK"
        )

        print(
            f"  Embeddings: {embeddings.shape}"
        )

        print(
            f"  Patches   : {len(patch_names)}"
        )


    # ========================================================
    # HANDLE CORRUPTED / INVALID FILE
    # ========================================================

    except Exception as e:

        print(
            f"  FAILED"
        )

        print(
            f"  Error: {type(e).__name__}: {e}"
        )


        failed_files.append({

            "file": str(wsi_file),

            "wsi": wsi_name,

            "error_type": type(e).__name__,

            "error": str(e)

        })


        continue


# ============================================================
# LOADING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LOADING SUMMARY")
print("=" * 70)

print(
    f"Total WSI files     : {len(wsi_files)}"
)

print(
    f"Successfully loaded : {successful_files}"
)

print(
    f"Failed files        : {len(failed_files)}"
)

print(
    f"Valid patches       : {total_patches}"
)


# ============================================================
# STOP IF NOTHING WAS LOADED
# ============================================================

if len(all_embeddings) == 0:

    raise RuntimeError(
        "\nNo valid WSI files were loaded. "
        "Clustering cannot continue."
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
        failed_path
    )


# ============================================================
# MERGE ALL EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("MERGING EMBEDDINGS")
print("=" * 70)


all_embeddings = np.concatenate(
    all_embeddings,
    axis=0
)


print(
    f"Final embeddings shape: "
    f"{all_embeddings.shape}"
)


print(
    f"Embedding names       : "
    f"{len(embedding_names)}"
)


print(
    f"Metadata entries      : "
    f"{len(patch_metadata)}"
)


# ============================================================
# CHECK CONSISTENCY
# ============================================================

if len(all_embeddings) != len(embedding_names):

    raise RuntimeError(
        "Number of embeddings does not match "
        "number of embedding names."
    )


if len(all_embeddings) != len(patch_metadata):

    raise RuntimeError(
        "Number of embeddings does not match "
        "number of metadata entries."
    )


# ============================================================
# EMBEDDING DIMENSION
# ============================================================

num_embeddings = all_embeddings.shape[0]

embedding_dim = all_embeddings.shape[1]


print(
    f"\nNumber of embeddings: {num_embeddings}"
)

print(
    f"Embedding dimension : {embedding_dim}"
)


# ============================================================
# L2 NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("L2 NORMALIZATION")
print("=" * 70)


X = normalize(
    all_embeddings,
    norm="l2"
).astype(
    np.float32
)


print(
    f"Normalized shape: {X.shape}"
)


# ============================================================
# TRAIN FAISS KMEANS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FAISS KMEANS")
print("=" * 70)


d = X.shape[1]


print(
    f"Dimension : {d}"
)

print(
    f"Clusters  : {K}"
)

print(
    f"Iterations: {NITER}"
)

print(
    f"GPU       : {USE_GPU}"
)


# ------------------------------------------------------------
# Check K
# ------------------------------------------------------------

if K <= 1:

    raise ValueError(
        "K must be greater than 1."
    )


if K > num_embeddings:

    raise ValueError(
        f"K ({K}) cannot be greater than "
        f"number of embeddings ({num_embeddings})."
    )


# ------------------------------------------------------------
# Create KMeans
# ------------------------------------------------------------

kmeans = faiss.Kmeans(

    d=d,

    k=K,

    niter=NITER,

    verbose=True,

    gpu=USE_GPU

)


print("\nTraining KMeans...")


kmeans.train(X)


print(
    "\nKMeans training completed."
)


# ============================================================
# CLUSTER CENTERS
# ============================================================

centers = kmeans.centroids


print(
    f"\nCluster centers shape: "
    f"{centers.shape}"
)


# ============================================================
# ASSIGN EMBEDDINGS TO CLUSTERS
# ============================================================

print("\n" + "=" * 70)
print("ASSIGNING EMBEDDINGS TO CLUSTERS")
print("=" * 70)


distances, labels = kmeans.index.search(
    X,
    1
)


labels = labels.flatten()

distances = distances.flatten()


print(
    f"Labels shape: {labels.shape}"
)


# ============================================================
# CLUSTER COUNTS
# ============================================================

cluster_counts = np.bincount(
    labels,
    minlength=K
)


print("\nCluster counts:")


for cluster_id in range(K):

    print(
        f"  Cluster {cluster_id}: "
        f"{cluster_counts[cluster_id]} embeddings"
    )


# ============================================================
# SAVE FAISS INDEX
# ============================================================

print("\n" + "=" * 70)
print("SAVING CLUSTERING")
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
# SAVE EMBEDDING -> CLUSTER MAPPING
# ============================================================

mapping_path = (
    OUTPUT_FOLDER /
    "embedding_cluster_mapping.npz"
)


np.savez(

    mapping_path,

    labels=labels.astype(
        np.int32
    ),

    distances=distances.astype(
        np.float32
    ),

    embedding_names=np.array(
        embedding_names,
        dtype=object
    ),

    patch_metadata=np.array(
        patch_metadata,
        dtype=object
    )

)


print(
    f"\nSaved embedding-cluster mapping:"
)

print(
    f"  {mapping_path}"
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "input_folder": str(
        INPUT_FOLDER
    ),

    "embedding_dimension": int(
        embedding_dim
    ),

    "number_of_embeddings": int(
        num_embeddings
    ),

    "number_of_clusters": int(
        K
    ),

    "kmeans_iterations": int(
        NITER
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
        successful_files
    ),

    "num_wsi_files_failed": int(
        len(failed_files)
    ),

    "cluster_counts": (
        cluster_counts.tolist()
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


print(
    f"\nSaved configuration:"
)

print(
    f"  {config_path}"
)


# ============================================================
# EXAMPLE EMBEDDING
# ============================================================

idx = 0


print("\n" + "=" * 70)
print("EXAMPLE EMBEDDING")
print("=" * 70)


print(
    "Embedding name:"
)

print(
    f"  {embedding_names[idx]}"
)


print(
    "\nCluster:"
)

print(
    f"  {labels[idx]}"
)


print(
    "\nDistance:"
)

print(
    f"  {distances[idx]}"
)


print(
    "\nMetadata:"
)

print(
    f"  {patch_metadata[idx]}"
)


# ============================================================
# SHOW EXAMPLES FROM EACH CLUSTER
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER EXAMPLES")
print("=" * 70)


for cluster_id in range(K):

    indices = np.where(
        labels == cluster_id
    )[0]


    print(
        f"\nCluster {cluster_id}: "
        f"{len(indices)} embeddings"
    )


    # Show first 5 examples
    for idx in indices[:5]:

        metadata = patch_metadata[idx]


        print(
            f"  {embedding_names[idx]} "
            f"(WSI={metadata['wsi']}, "
            f"patch={metadata['patch']})"
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
    f"WSI files loaded    : {successful_files}"
)

print(
    f"WSI files failed    : {len(failed_files)}"
)

print(
    f"Total embeddings    : {num_embeddings}"
)

print(
    f"Embedding dimension : {embedding_dim}"
)

print(
    f"Number of clusters  : {K}"
)


print("\nCluster distribution:")


for cluster_id in range(K):

    percentage = (
        cluster_counts[cluster_id]
        / num_embeddings
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

print(
    f"  3. {mapping_path.name}"
)

print(
    f"  4. {config_path.name}"
)

if failed_files:

    print(
        f"  5. failed_files.json"
    )


print(
    f"\nOutput directory:"
)

print(
    f"  {OUTPUT_FOLDER}"
)

print("\nDone.")