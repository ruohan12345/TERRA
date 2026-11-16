#!/usr/bin/env python3
"""Chinese illustrations of balanced window ownership and one periodic shift.

This is an analytical illustration, not a training or hardware benchmark.
Run with a Python environment containing numpy and matplotlib:
    python -B render_layouts.py --output-dir figures
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import colors as mcolors
from matplotlib import font_manager

RANK_COLORS = ['#2875B8', '#E08A33', '#449879', '#9871AE',
               '#D36779', '#8F8059', '#5F9BB8', '#65758B']


def setup_font():
    requested = os.environ.get('TERRA_PLOT_FONT')
    candidates = [requested, 'C:/Windows/Fonts/msyh.ttc',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                  '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']
    for name in candidates:
        if name and Path(name).is_file():
            if hasattr(font_manager.fontManager, 'addfont'):
                font_manager.fontManager.addfont(name)
            else:
                font_manager.fontManager.ttflist.extend(font_manager.createFontList([name]))
            prop = font_manager.FontProperties(fname=name)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['svg.fonttype'] = 'path'
            return name
    raise RuntimeError('Chinese font required; set TERRA_PLOT_FONT to its file path')


def ownership(rows, columns, ranks, order, cut_policy):
    n = rows * columns
    if cut_policy == 'paper_floor':
        cuts = [p * n // ranks for p in range(ranks + 1)]
    else:
        q, remainder = divmod(n, ranks)
        cuts = [p * q + min(p, remainder) for p in range(ranks + 1)]
    owner = np.empty((rows, columns), dtype=np.int64)
    for rank in range(ranks):
        for index in range(cuts[rank], cuts[rank + 1]):
            if order == 'row':
                i, j = divmod(index, columns)
            else:
                j, i = divmod(index, rows)
            owner[i, j] = rank
    return owner, cuts


def token_plan(owner, ranks, window, shift):
    pixels = np.repeat(np.repeat(owner, window, axis=0), window, axis=1)
    height, width = pixels.shape
    mask = np.empty((height, width), dtype=np.bool_)
    matrix = np.zeros((ranks, ranks), dtype=np.int64)
    # Every source token is visited once. Shift and wrap are explicit here.
    for y in range(height):
        for x in range(width):
            source = int(pixels[y, x])
            target = int(pixels[(y-shift) % height, (x-shift) % width])
            matrix[source, target] += 1
            mask[y, x] = source != target
    assert int(matrix.sum()) == height * width
    assert int(mask.sum()) == int(matrix.sum() - np.trace(matrix))
    return pixels, mask, matrix


def lighten(color, strength):
    color = np.array(mcolors.to_rgb(color))
    return (1.0-strength) * np.ones(3) + strength * color


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0, labelsize=11)
    ax.set_aspect('equal')


def save_figure(fig, stem, out):
    for suffix in ('.png', '.svg'):
        fig.savefig(str(out / (stem + suffix)), dpi=170, facecolor='white')
    plt.close(fig)


def draw_owners(layouts, args, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
    fig.subplots_adjust(left=.055, right=.985, bottom=.23, top=.76, wspace=.12)
    fig.suptitle('同一窗口网格，改变分配方向', fontsize=21, y=.965)
    fig.text(.5, .87, '示意网格：{}×{} 个窗口 · {} 个 rank · 窗口 ID 始终按行编号'.format(
        args.rows, args.columns, args.ranks), ha='center', fontsize=12, color='#4D5966')
    titles = ['行优先：沿窗口行扫描分配', '列优先：沿窗口列扫描分配']
    for ax, (mode, (owner, cuts)), title in zip(axes, layouts.items(), titles):
        for i in range(args.rows):
            for j in range(args.columns):
                rank = int(owner[i, j])
                ax.add_patch(patches.Rectangle((j, i), 1, 1,
                    facecolor=lighten(RANK_COLORS[rank], .36),
                    edgecolor='white', linewidth=2))
                ax.text(j+.5, i+.5, str(i*args.columns+j), ha='center', va='center',
                        fontsize=13, color='#182B3C')
        ax.set_xlim(0, args.columns)
        ax.set_ylim(args.rows, 0)
        ax.set_xticks(np.arange(args.columns)+.5)
        ax.set_xticklabels([str(j) for j in range(args.columns)])
        ax.set_yticks(np.arange(args.rows)+.5)
        ax.set_yticklabels([str(i) for i in range(args.rows)])
        ax.set_xlabel('窗口列', fontsize=12, labelpad=7)
        ax.set_ylabel('窗口行', fontsize=12, labelpad=7)
        ax.set_title(title, fontsize=15, pad=15)
        style_axes(ax)
    handles = [patches.Patch(facecolor=lighten(RANK_COLORS[p], .36),
               label='rank {}'.format(p), edgecolor='none') for p in range(args.ranks)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.115),
               ncol=args.ranks, frameon=False, fontsize=12, handlelength=1.3)
    fig.text(.5, .045, '颜色表示窗口归属；两侧窗口数、位置与全局编号完全相同。',
             ha='center', fontsize=12, color='#4D5966')
    save_figure(fig, 'owner_layouts', out)


def draw_traffic(plans, args, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.05))
    fig.subplots_adjust(left=.055, right=.985, bottom=.245, top=.735, wspace=.12)
    fig.suptitle('一次内部移窗：哪些 token 需要跨 rank？', fontsize=21, y=.965)
    fig.text(.5, .87, '同向双轴 shift：向上 {} 格、向左 {} 格，越界后周期回绕'.format(
        args.shift, args.shift), ha='center', fontsize=12, color='#4D5966')
    height, width = args.rows*args.window_size, args.columns*args.window_size
    for ax, (mode, (pixels, mask, matrix)) in zip(axes, plans.items()):
        rgb = np.empty((height, width, 3), dtype=np.float64)
        for rank in range(args.ranks):
            selected = pixels == rank
            rgb[selected & ~mask] = lighten(RANK_COLORS[rank], .14)
            rgb[selected & mask] = lighten(RANK_COLORS[rank], .86)
        ax.imshow(rgb, origin='upper', extent=(0, width, height, 0), interpolation='none')
        # Thin token grid; thicker lines delimit complete attention windows.
        for x in range(width+1):
            ax.axvline(x, color='white' if x % args.window_size else '#65717D',
                       linewidth=.35 if x % args.window_size else .95, alpha=.8)
        for y in range(height+1):
            ax.axhline(y, color='white' if y % args.window_size else '#65717D',
                       linewidth=.35 if y % args.window_size else .95, alpha=.8)
        remote = int(mask.sum()); total = int(mask.size)
        title = '行优先' if mode == 'row' else '列优先'
        ax.set_title('{}：{} / {} 个 token 跨 rank（{:.0f}%）'.format(
            title, remote, total, 100.*remote/total), fontsize=14.2, pad=15)
        ax.set_xticks(np.arange(0, width+1, args.window_size))
        ax.set_yticks(np.arange(0, height+1, args.window_size))
        ax.set_xlabel('token 横坐标', fontsize=12, labelpad=7)
        ax.set_ylabel('token 纵坐标', fontsize=12, labelpad=7)
        # One token illustrates the displacement; the mask itself is exact.
        if height >= 8 and width >= 12 and args.shift > 0:
            source = (2*args.window_size+1.5, args.window_size+1.5)
            target = (source[0]-args.shift, source[1]-args.shift)
            ax.plot([source[0]], [source[1]], marker='o', markersize=4,
                    markerfacecolor='white', markeredgecolor='#17212B', zorder=8)
            ax.annotate('', xy=target, xytext=source,
                        arrowprops=dict(arrowstyle='->', color='#17212B', lw=1.65), zorder=8)
        style_axes(ax)
    handles = [patches.Patch(facecolor='#F0F2F4', edgecolor='#AEB7C0', label='浅色：留在本卡'),
               patches.Patch(facecolor='#52677D', edgecolor='none', label='深色：需要 P2P 发送')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.115),
               ncol=2, frameon=False, fontsize=12, handlelength=1.5)
    fig.text(.5, .06, '每个小格是一个源 token；粗线为窗口边界，箭头示意一次平移。',
             ha='center', fontsize=11.5, color='#4D5966')
    fig.text(.5, .022, '统计完整周期域的单次逻辑通信量，包含首尾回绕；不含入口／出口转换。',
             ha='center', fontsize=11, color='#4D5966')
    save_figure(fig, 'shift_traffic', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--columns', type=int, default=8)
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--window-size', type=int, default=4)
    parser.add_argument('--shift', type=int, default=2)
    parser.add_argument('--cut-policy', choices=('paper_floor','prefix_remainder'), default='paper_floor')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'figures')
    args = parser.parse_args()
    if min(args.rows,args.columns,args.ranks,args.window_size)<1 or args.ranks>8:
        parser.error('Positive dimensions and at most eight rank colors required')
    if args.rows*args.columns>160 or args.rows*args.window_size>96 or args.columns*args.window_size>128:
        parser.error('Use a small illustrative grid; this renderer is for readable diagrams')
    if not 0<=args.shift<args.window_size:
        parser.error('shift must be in [0, window_size)')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = setup_font()
    layouts = {mode:ownership(args.rows,args.columns,args.ranks,mode,args.cut_policy)
               for mode in ('row','column')}
    plans = {mode:token_plan(owner,args.ranks,args.window_size,args.shift)
             for mode,(owner,cuts) in layouts.items()}
    draw_owners(layouts,args,args.output_dir)
    draw_traffic(plans,args,args.output_dir)
    data = {'kind':'analytical_layout_and_token_route_illustration_no_hardware_timing',
            'parameters':{'rows':args.rows,'columns':args.columns,'ranks':args.ranks,
              'window_size':args.window_size,'shift_y':args.shift,'shift_x':args.shift,
              'cut_policy':args.cut_policy,'global_window_id_rule':'i * columns + j'},
            'Chinese_font':Path(font).name,'layouts':{},'files':[]}
    for mode,(owner,cuts) in layouts.items():
        pixels,mask,matrix=plans[mode]
        data['layouts'][mode]={'window_owner_grid':owner.tolist(),'cuts':cuts,
            'send_matrix':matrix.tolist(),'remote_tokens':int(mask.sum()),
            'total_tokens':int(mask.size),'remote_source_token_mask':mask.astype(int).tolist()}
    for p in sorted(args.output_dir.glob('*')):
        if p.is_file() and p.suffix in ('.png','.svg'):
            data['files'].append({'path':p.name,'bytes':p.stat().st_size,
                                 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    receipt=Path(__file__).resolve().parent/'FIGURE_RECEIPT.json'
    receipt.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'GENERATED','remote_tokens':{m:v['remote_tokens'] for m,v in data['layouts'].items()},
                      'figures':len(data['files']),'receipt':str(receipt)},ensure_ascii=False))


if __name__ == '__main__':
    main()
