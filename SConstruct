from SCons.Script import VariantDir, Environment, Builder, Depends, Flatten
import os
import glob

VariantDir('_build', src_dir='.')

env = Environment(ENV=os.environ)
inkscape = Builder(action = 'inkscape --without-gui --export-pdf=$TARGET $SOURCE')
env['BUILDERS']['Inkscape'] = inkscape
env['BUILDERS']['Latexdiff'] = Builder(action = 'latexdiff $SOURCES > $TARGET')

converted_pdfs = [env.Inkscape(target="figures/" + os.path.basename(svg).replace('.svg','.pdf'), source=svg)
               for svg in glob.glob('svg-fig/*.svg')]

joint_inf, = env.PDF(target='_build/joint_inf.pdf',source='joint_inf.tex')

env.Latexdiff(target='diff.tex',source=['versions/joint_inf.v2.tex','joint_inf.tex'])
diff = env.PDF(target='diff.pdf',source='diff.tex')

Depends(Flatten([joint_inf]),
        Flatten([converted_pdfs, 'joint_inf.bib']))

cont_build = env.Command('.continuous', ['joint_inf.bib', 'joint_inf.tex', 'appendix.tex'],
    'while :; do inotifywait -e modify $SOURCES; scons -Q; done')
Alias('continuous', cont_build)

Default([joint_inf])
