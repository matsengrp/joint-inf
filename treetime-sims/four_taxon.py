import os
import numpy as np
from collections import defaultdict
from Bio import AlignIO,Phylo
from treetime.treeanc import TreeAnc
from treetime.gtr import GTR
from treetime.seqgen import SeqGen
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

def invFels(y,x, L=1000, alphabet='nuc_nogap'):
    from io import StringIO
    T = Phylo.read(StringIO("(A:%f,B:%f,(C:%f,D:%f):%f);"%(y,y,x,x,y)), "newick")

    gtr = GTR(alphabet=alphabet)
    gtr.seq_len=L
    mySeq = SeqGen(gtr=gtr, tree=T)
    mySeq.evolve()

    return T, mySeq.get_aln()

def tiny_internal(T):
    T.root.clades[2].branch_length = 0.0001
    T.root.clades[2].mutation_length = 0.0001

def noisy_shortened(T):
    for n in T.find_clades():
        n.branch_length *= 0.6 + np.random.random()*0.4
        n.mutation_length=n.branch_length

def topology_only(T):
    for n in T.find_clades():
        n.branch_length = 0.1
        n.mutation_length=n.branch_length

def infer_branch_length(tree, aln, distance_scale = 1.0,
                        marginal=False, IC=None, alphabet='nuc_nogap'):
    before = {}
    after = {}
    tt = TreeAnc(tree=tree, aln=aln, alphabet=alphabet, verbose=0)
    for n in tt.tree.find_clades():
        n.branch_length   *= distance_scale
        n.mutation_length *= distance_scale
        before[n.name] = n.branch_length
    total_before = tt.tree.total_branch_length()


    # mess up branch length prior to optimizition
    if IC:
        IC(tt.tree)
    tt.prepare_tree()

    if marginal:
        tt.optimize_sequences_and_branch_length(branch_length_mode='marginal',
                                                prune_short=False, max_iter=20)
    else:
        tt.optimize_sequences_and_branch_length(branch_length_mode='joint',
                                                prune_short=False, max_iter=20)

    for n in tt.tree.find_clades():
        after[n.name] = n.branch_length
    total_after = tt.tree.total_branch_length()

    LH = tt.sequence_LH() if marginal else tt.tree.sequence_joint_LH

    return np.array([[before[k], after[k]] for k in before]), LH


def plot_branch_length(bl, l, suffix, c=10):
    x_grid = np.repeat([l], len(l), axis=0)
    y_grid = np.copy(x_grid.T)

    total_bl = bl.sum(axis=2)
    tbl = (2*x_grid+3*y_grid)

    fig, axs = plt.subplots(1,2, sharey=True, figsize=(15,6))
    axs[0].set_title("absolute total branch length deviation")
    sns.heatmap(total_bl[:,:,1]-total_bl[:,:,0], ax=axs[0], square=True, vmin=-.1, vmax=.1, cmap='bwr')
    axs[0].invert_yaxis()
    axs[0].set_xticklabels(["%1.4f"%v for v in l],rotation=60)
    axs[0].set_yticklabels(["%1.4f"%v for v in l],rotation=0)

    axs[1].set_title("relative total branch length deviation")
    sns.heatmap(total_bl[:,:,1]/total_bl[:,:,0]-1, ax=axs[1], square=True, vmin=-1, vmax=1, cmap='bwr')
    axs[1].invert_yaxis()
    axs[1].set_xticklabels(["%1.4f"%v for v in l],rotation=60)
    axs[1].set_yticklabels(["%1.4f"%v for v in l], rotation=90)

    plt.tight_layout()
    plt.savefig("invFels_total_branch_deviation_heatmap"+suffix)


    fig, axs = plt.subplots(1,2, sharey=True, figsize=(15,6))
    axs[0].set_title("absolute internal branch length deviation")
    sns.heatmap(bl[:,:,3,1]-bl[:,:,3,0], ax=axs[0], square=True, vmin=-.1, vmax=.1, cmap='bwr')
    axs[0].invert_yaxis()
    axs[0].set_xticklabels(["%1.4f"%v for v in l],rotation=60)
    axs[0].set_yticklabels(["%1.4f"%v for v in l],rotation=0)

    axs[1].set_title("relative internal branch length deviation")
    sns.heatmap(bl[:,:,3,1]/bl[:,:,3,0]-1, ax=axs[1], square=True, vmin=-1, vmax=1, cmap='bwr')
    axs[1].invert_yaxis()
    axs[1].set_xticklabels(["%1.4f"%v for v in l],rotation=60)
    axs[1].set_yticklabels(["%1.4f"%v for v in l], rotation=90)

    plt.tight_layout()
    plt.savefig("invFels_internal_branch_deviation_heatmap"+suffix)


    fig, axs = plt.subplots(1,2, figsize=(15,6))
    axs[0].plot(tbl.flatten(), total_bl[:,:,1].flatten(), 'o', label="joint optimization")
    axs[0].plot([0,tbl.max()],[0,tbl.max()])
    x = np.linspace(0,tbl.max())
    axs[0].plot(x,x*(1-x/5/c))
    axs[0].set_ylabel(r"inferred total branch length $\ell$")
    axs[0].set_xlabel("total branch length")
    axs[0].legend()

    axs[1].plot(tbl.flatten()/5, (total_bl[:,:,1]/total_bl[:,:,0]-1).flatten(), 'o', label="joint optimization")
    axs[1].plot(x/5,-x/5/c, label="$x/%f$"%c)
    axs[1].set_ylabel("relative deviation of total branch length")
    axs[1].set_xlabel("average branch length")
    plt.tight_layout()
    plt.savefig("invFels_tbl_deviation"+suffix)

    fig, axs = plt.subplots(1,2, figsize=(15,6))
    axs[0].scatter(y_grid.flatten(), bl[:,:,3,1].flatten(), label="joint optimization", c=(x_grid**2/4>y_grid+y_grid**2/2).flatten())
    axs[0].plot([0,y_grid.max()],[0,y_grid.max()])
    x = np.linspace(0,y_grid.max())
    axs[0].set_ylabel("inferred internal branch length")
    axs[0].set_xlabel(r"true length $y$")
    axs[0].legend()

    axs[1].scatter(tbl.flatten(), (bl[:,:,3,1]/bl[:,:,3,0]-1).flatten(), label="joint optimization")
    axs[1].set_ylabel("relative deviation of internal branch length")
    axs[1].set_xlabel(r"total_branch_length")
    plt.tight_layout()
    plt.savefig("invFels_internal_branch_deviation"+suffix)


    fig, axs = plt.subplots(1,2, sharey=True, figsize=(15,6))
    axs[0].set_title(r"$p_x^2 > 2p_y$")
    sns.heatmap(x_grid**2/2-y_grid**2/2>y_grid, ax=axs[0], square=True)
    axs[0].invert_yaxis()
    axs[0].set_xticklabels(["%1.4f"%v for v in l],rotation=60)
    axs[0].set_yticklabels(["%1.4f"%v for v in l],rotation=0)

    axs[1].set_title("relative internal branch length deviation")
    sns.heatmap(bl[:,:,3,1]/bl[:,:,3,0]-1, ax=axs[1], square=True, vmin=-1, vmax=1, cmap='bwr')
    axs[1].invert_yaxis()
    axs[1].set_xticklabels(["%1.4f"%v for v in l],rotation=60)
    axs[1].set_yticklabels(["%1.4f"%v for v in l], rotation=90)

    plt.tight_layout()
    plt.savefig("invFels_topology_region_heatmap"+suffix)

if __name__ == '__main__':
    L=30000
    marginal=False
    LH_all = defaultdict(list)
    bl_all = defaultdict(list)
    alphabet = np.array(['A', 'G'])
    #alphabet = 'nuc_nogap'
    methods =[(noisy_shortened, 'noisy')] #, (tiny_internal, 'tiny_internal')] #, (topology_only, 'topo')]
    l = np.logspace(-3,np.log10(0.5),21)
    #l = np.linspace(0.001, 0.1, 21)
    for x in l:
        r = defaultdict(list)
        lh = defaultdict(list)
        for y in l:
            T, aln = invFels(x, y, L=L, alphabet=alphabet)
            orig_bl = [n.branch_length for n in T.find_clades()]
            for (ic, fname) in methods:
                bl,tmp_lh = infer_branch_length(T, aln, marginal=marginal, IC=ic, alphabet=alphabet)
                r[fname].append(bl)
                lh[fname].append(tmp_lh)
                for b,n in zip(orig_bl, T.find_clades()):
                    n.branch_length=b
                    n.mutation_length=b


        for (ic, fname) in methods:
            bl_all[fname].append(r[fname])
            LH_all[fname].append(lh[fname])


    for (ic, fname) in methods:
        suffix = "_%s_joint_log%s.pdf"%(fname, "" if type(alphabet)==str else "_binary")
        bl_all[fname] = np.array(bl_all[fname])
        LH_all[fname] = np.array(LH_all[fname])

        plot_branch_length(bl_all[fname], l, suffix, c=2 if type(alphabet)==str else 1)
