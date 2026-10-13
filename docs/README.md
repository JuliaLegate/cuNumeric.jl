# Documentation preview

From `docs/`, run:

```sh
npm install
npm run docs:dev
```

Open <http://127.0.0.1:5173>. Edit Markdown, components, or theme files under
`docs/src/`; the preview reloads automatically. Edit `pages` in `docs/make.jl`
to change the navigation. Press Ctrl+C to stop the server.

This source preview works without loading Julia or cuNumeric, including on
Windows. It uses the site's existing VitePress theme and copies files into the
ignored `docs/build/.source-preview/` directory. It does not execute Julia
examples or expand Documenter API directives; those blocks are marked in the
preview. Julia cross-references also require the full build.

For the complete documentation, use the Julia environment and build steps in
`.github/workflows/docs.yml` on a supported Linux host. After
`julia --project=docs docs/make.jl`, run `npm run docs:dev:generated` from
`docs/` to serve the generated pages. That server watches the generated output;
rerun the Julia build to incorporate source changes.
