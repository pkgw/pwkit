# The pwkit release process

These are notes for the `pwkit` developers about how to create a new release.

1. Create a branch intended to become the next release.
2. Finish features, test functionality, etc.
3. `python setup.py sdist` and verify contents.
4. Make sure that `CHANGES.md` is up-to-date.
5. For the final commit, update the version number in `setup.py` and
   `docs/conf.py`, and add a proper version and date to `CHANGES.md`. Commit
   with message `Release version ${version}`.
6. Push to GitHub and create a pull request for the new release called
   "Release PR for version $version".
7. Get it so that it passes CI, creating fixup commits as necessary.
8. When it's really really ready, `git clean -fxd && python setup.py sdist &&
   twine upload dist/*.tar.gz`. If `twine` finds problems, make any final
   changes and retry.
9. If needed, do a `git rebase -i` to make the version-bump commit the last
   one again.
10. `git tag v${version}`
11. Update the version number to `${cur_major}.${next_minor}.0.dev0` and add a
    new separator in `CHANGES.md` along the lines of `${version} (unreleased)`.
    Commit with a message of `Back to development.`
12. `git push` (with `-f` if history was rewritten) to the PR branch. This had
    *really* better still pass CI.
13. Merge into `master`.
14. Pull the merged `master` locally.
15. `git push --tags`
16. Create a new release on GitHub and copy the latest contents of
    `CHANGES.md` into the description.
