from SCons.Script import VariantDir, Environment, \
        Builder, Depends, Flatten
import os

VariantDir('_build', src_dir='.')

env = Environment(ENV=os.environ)
inkscape = Builder(action = 'inkscape --without-gui --export-pdf=$TARGET $SOURCE')
env['BUILDERS']['Inkscape'] = inkscape
env['BUILDERS']['Latexdiff'] = Builder(action = 'latexdiff $SOURCES > $TARGET')

svgs = [
    'figures/topology-inconsistency-legend.svg',
    'figures/w-hat-heatmap-joint-diff-evol.svg',
    'figures/farris_like00.svg',
    'figures/farris_like01.svg',
    'figures/farris_like10.svg',
    'figures/farris_like11.svg',
    'figures/unrooted_four_taxa.svg',
    'figures/farris_blank.svg',
    'figures/felsenstein_blank.svg',
    'figures/branch-length-inconsistency-legend.svg',
    'figures/bl-loose-inconsistency.svg',
]

pdfs = [env.Inkscape(target="figures/" + os.path.basename(svg).replace('.svg','.pdf'), source=svg)
        for svg in svgs]

joint_inf, = env.PDF(target='_build/joint_inf.pdf',source='joint_inf.tex')
Default([joint_inf])

Depends(Flatten([joint_inf]),
        Flatten([pdfs, 'joint_inf.bib']))

cont_build = env.Command('.continuous', ['joint_inf.bib', 'joint_inf.tex'],
    'while :; do inotifywait -e modify $SOURCES; scons -Q; done')
Alias('continuous', cont_build)
