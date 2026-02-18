#!/bin/bash

# directory of input datasets
# |- 1.zarr
#   |- row
#     |- well
#       |- fov
# |- 2.zarr
# |- ...
input_datasets=$1

# template zarr store path, e.g. the phase input
template_store=$2

# output zarr store path
output_store=$3


for well in $template_store/*/*; do
    well_name=$(realpath --relative-to="$template_store" "$well")
    target_dir=$output_store/$well_name
    mkdir -p $target_dir
    # well metadata
    cp -t $target_dir $well/.z*
    # images
    mv -t $target_dir $input_datasets/*/$well_name/*
done

# plate metadata
cp -t $output_store $template_store/.z*

#add zgroup to row level
for row in $template_store/*; do
    cp -t $output_store/$(basename $row) $row/.zgroup
done

#Cleanup
# rm -r $input_datasets