mkdir pages
python docs/create_base_html.py
python run.py install docs
mv docs/_build/html/ pages/latest/

for tag in $(git tag); do
    echo "Processing tag: $tag"
    git checkout "$tag"
    python -m pip install .
    git checkout main -- docs/conf.py
    python run.py docs
    mv docs/_build/html/ "pages/$tag/"
    echo "Finished processing tag: $tag"
done

git checkout main
