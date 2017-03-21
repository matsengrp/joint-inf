from SCons.Script import VariantDir, Environment,         Builder, Depends, Flatten
import os

VariantDir('_build', src_dir='.')

env = Environment(ENV=os.environ)
env['BUILDERS']['Latexdiff'] = Builder(action = 'latexdiff $SOURCES > $TARGET')

joint_inf, = env.PDF(target='_build/joint_inf.pdf',source='joint_inf.tex')
#joint_inf_supp, = env.PDF(target='_build/joint_inf_supp.pdf', source='joint_inf_supp.tex')
Default([joint_inf])

#env.Latexdiff(target='diff.tex',source=['stored_joint_inf.tex','joint_inf.tex'])
#diff = env.PDF(target='diff.pdf',source='diff.tex')

Depends(Flatten([joint_inf]),
        Flatten(['joint_inf.bib'])) #, 'defs.tex'

#Depends(joint_inf, joint_inf_supp)

cont_build = env.Command('.continuous', ['joint_inf.bib', 'joint_inf.tex'],
    'while :; do inotifywait -e modify $SOURCES; scons -Q; done')
Alias('continuous', cont_build)
