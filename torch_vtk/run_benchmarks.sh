python preprocessing/converters/torch_to_hdf5.py /run/media/dome/Data/data/Volumes/tvdls /run/media/dome/Data/data/Volumes/tvdls_lzf.h5 -compression lzf
python preprocessing/converters/torch_to_hdf5.py /run/media/dome/Data/data/Volumes/tvdls /run/media/dome/Data/data/Volumes/tvdls_gzip.h5 -compression gzip
python benchmarks/benchmark_cq500_hdf5.py /run/media/dome/Data/data/Volumes/tvdls_lzf.h5 -plot ~/Pictures/tvdls_hdf5_lzf_results.png
python benchmarks/benchmark_cq500_hdf5.py /run/media/dome/Data/data/Volumes/tvdls_gzip.h5 -plot ~/Pictures/tvdls_hdf5_gzip_results.png
