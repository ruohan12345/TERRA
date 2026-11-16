"""Chinese hotspot charts from preclassified statistics, never reparse traces.

Actual task elapsed sums and communication envelope sums use different
denominators. Neither is an additive partition of the optimizer-step wall.
"""
import math
from pathlib import Path
import unicodedata

from .figures import configure_matplotlib_font,device_technology,_context_label


def _category(task_kind):
    if task_kind=='gpu_compute':
        return 'compute'
    if task_kind in ('gpu_transport','gpu_comm'):
        return 'communication'
    return 'copy_or_unknown'


def build_hotspot_panel(hotspots,device,table_kind,phases,limit):
    """Merge phases for display only. Full source rows/totals remain intact."""
    if not isinstance(limit,int) or isinstance(limit,bool) or limit<1:
        raise ValueError('Hotspot limit must be a positive integer')
    selected=[r for r in hotspots.get('rows',[]) if r['device']==device and r['table_kind']==table_kind and r['phase'] in phases]
    totals=[r for r in hotspots.get('totals',[]) if r['device']==device and r['table_kind']==table_kind and r['phase'] in phases]
    denominator=sum(r['denominator_ns'] for r in totals)
    if not math.isfinite(denominator) or denominator<0:
        raise ValueError('Invalid hotspot denominator')
    grouped={}
    category_totals={key:0 for key in ('compute','communication','copy_or_unknown')}
    for row in selected:
        value=row['duration_sum_ns']
        if not math.isfinite(value) or value<0:
            raise ValueError('Invalid hotspot duration')
        key=(row.get('display_name',row['name']),row['task_kind'])
        group=grouped.setdefault(key,{'name':key[0],'task_kind':key[1],'count':0,'duration_sum_ns':0,'source_phases':set()})
        group['count']+=row['count'];group['duration_sum_ns']+=value;group['source_phases'].add(row['phase'])
        category_totals[_category(row['task_kind'])]+=value
    accumulated=sum(r['duration_sum_ns'] for r in grouped.values())
    if abs(accumulated-denominator)>max(1.,denominator*1e-9):
        raise ValueError('Hotspot rows disagree with their table/phase denominator')
    records=sorted(grouped.values(),key=lambda r:(-r['duration_sum_ns'],r['name'],r['task_kind']))
    top=[]
    for row in records[:limit]:
        top.append({**row,'source_phases':sorted(row['source_phases']),
                    'share_percent':100*row['duration_sum_ns']/denominator if denominator else None})
    remainder=sum(r['duration_sum_ns'] for r in records[limit:])
    if remainder>0:
        top.append({'name':'其余已归因任务','task_kind':'other_mixed','count':sum(r['count'] for r in records[limit:]),
                    'duration_sum_ns':remainder,'source_phases':list(phases),'share_percent':100*remainder/denominator})
    return {'device':device,'table_kind':table_kind,'phases':list(phases),'denominator_ns':denominator,
            'source_row_count':len(selected),'grouped_row_count':len(records),'top_rows':top,
            'category_duration_sum_ns':category_totals,
            'sum_display_percent':sum(r['share_percent'] for r in top) if denominator else None,
            'denominator_scope':{'device_tasks':'sum of actual device-task elapsed times; overlaps allowed',
                                 'communication_kernels':'sum of actual GPU communication-kernel elapsed times; overlaps allowed',
                                 'communication_envelopes':'sum of communication-envelope elapsed times, including waits'}[table_kind]}


def build_communication_panel(hotspots,device,phases,limit,technology):
    """Native envelope or CUDA communication-kernel subset, for display only."""
    if technology!='GPU':
        return build_hotspot_panel(hotspots,device,'communication_envelopes',phases,limit)
    envelopes=[r for r in hotspots.get('rows',[]) if r['device']==device and r['table_kind']=='communication_envelopes' and r['phase'] in phases]
    if envelopes:
        return build_hotspot_panel(hotspots,device,'communication_envelopes',phases,limit)
    rows=[{**r,'table_kind':'communication_kernels'} for r in hotspots.get('rows',[])
          if r['device']==device and r['table_kind']=='device_tasks' and r['task_kind']=='gpu_comm' and r['phase'] in phases]
    totals={}
    for row in rows:
        totals[row['phase']]=totals.get(row['phase'],0)+row['duration_sum_ns']
    projection={'rows':rows,'totals':[{'table_kind':'communication_kernels','device':device,'phase':phase,'denominator_ns':value}
                                     for phase,value in totals.items()]}
    return build_hotspot_panel(projection,device,'communication_kernels',phases,limit)


def _short(text,limit=37):
    text=str(text)
    width=lambda letter:2 if unicodedata.east_asian_width(letter) in ('F','W') else 1
    if sum(width(letter) for letter in text)<=limit:return text
    def take(letters,budget):
        output='';used=0
        for letter in letters:
            if used+width(letter)>budget:break
            output+=letter;used+=width(letter)
        return output
    # Preserve both the API prefix and distinguishing kernel suffix. The
    # unabridged identifier remains in canonical JSON/CSV and panel metadata.
    head=(limit-1)//2
    return take(text,head)+'…'+take(reversed(text),limit-head-1)[::-1]


def render_hotspot_figures(bundle,analysis,hotspots,prefix,*,top_n=10,communication_top_n=5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    font=configure_matplotlib_font()
    prefix=Path(prefix);paths=[];plots=[]
    devices=sorted({r['device'] for r in hotspots.get('totals',[]) if r['table_kind'] in ('device_tasks','communication_envelopes') and r['device']!='cpu'})
    for device_index,device in enumerate(devices):
        technology=device_technology(bundle.metadata,device,getattr(bundle,'device_adapter',{}))
        for name,phases,title in [('forward',('forward',),'模型前向'),
                                  ('backward',('backward','gradient_finalize'),'反向与梯度规约收尾')]:
            actual=build_hotspot_panel(hotspots,device,'device_tasks',phases,top_n)
            envelope=build_communication_panel(hotspots,device,phases,communication_top_n,technology['type'])
            fig,axes=plt.subplots(1,2,figsize=(16,8.6),gridspec_kw={'width_ratios':[1.12,1]})
            fig.subplots_adjust(left=.20,right=.94,top=.73,bottom=.26,wspace=.92)
            for panel,(ax,data) in enumerate(zip(axes,(actual,envelope))):
                rows=data['top_rows']
                if not rows or not data['denominator_ns']:
                    ax.axis('off');ax.text(.5,.5,'未观察到／无可归属的记录\n不能据此认为成本为零',ha='center',va='center',transform=ax.transAxes,fontsize=12,color='#687A8A')
                else:
                    for index,row in enumerate(rows):
                        category=_category(row['task_kind'])
                        color=('#427EAA' if category=='compute' else '#E39B43' if category=='communication' else '#9BA8B5') if panel==0 else '#E39B43'
                        ax.barh(index,row['share_percent'],color=color,height=.62,hatch='//' if row['task_kind']=='other_mixed' else None)
                        ax.text(row['share_percent']+.8,index,'%.1f%%  /  %.2f ms' % (row['share_percent'],row['duration_sum_ns']/1e6),va='center',fontsize=9)
                    ax.set_yticks(range(len(rows)),[_short(r['name']) for r in rows],fontsize=9)
                    ax.invert_yaxis();maximum=max(r['share_percent'] for r in rows)
                    ax.set_xlim(0,max(12,maximum*1.52));ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
                    ax.set_xlabel('占本栏累计耗时（%）')
                heading=('实际'+technology['label']+'任务热点' if panel==0 else
                         'GPU 通信内核热点' if data['table_kind']=='communication_kernels' else
                         'HCOM 通信包络热点（含等待）' if technology['type']=='NPU' else '通信包络热点（含等待）')
                scope=('本栏分母 %.3f ms' % (data['denominator_ns']/1e6)) if data['denominator_ns'] else '本栏无可归属记录'
                ax.set_title(heading+'\n'+scope,fontsize=12,loc='left',pad=12)
            fig.text(.035,.95,title+'：热点算子与通信',fontsize=20,weight='bold')
            fig.text(.035,.87,_context_label(bundle,analysis['selected_update'])+' · '+device,fontsize=10.5,linespacing=1.4)
            fig.legend(handles=[Patch(facecolor='#427EAA',label='实际计算任务'),Patch(facecolor='#E39B43',label='真实通信传输／通信kernel'),
                                Patch(facecolor='#9BA8B5',label='显式拷贝／其他任务')],loc='lower left',bbox_to_anchor=(.03,.177),ncol=3,frameon=False,fontsize=10)
            phase_total=sum(r['denominator_ns'] for r in hotspots.get('totals',[]) if r['device']==device and r['table_kind']=='device_tasks')
            unknown=sum(r['denominator_ns'] for r in hotspots.get('totals',[]) if r['device']==device and r['table_kind']=='device_tasks' and r['phase']=='unknown')
            comm=actual['category_duration_sum_ns']['communication']
            comm_share=100*comm/actual['denominator_ns'] if actual['denominator_ns'] else None
            fig.text(.035,.142,'左栏真实通信任务累计 %.3f ms%s；本步阶段未知任务 %.3f ms / 累计 %.3f ms。' % (
                comm/1e6,'（%.1f%%）' % comm_share if comm_share is not None else '（比例未知）',unknown/1e6,phase_total/1e6),fontsize=10)
            scope='右：GPU 通信内核累计耗时（左栏通信子集，不可重复相加）。' if envelope['table_kind']=='communication_kernels' else '右：通信包络累计耗时（含等待）。两个分母独立，不可相加。'
            fig.text(.035,.101,'左：实际任务累计耗时；'+scope,fontsize=10)
            fig.text(.035,.061,'任务可以跨stream重叠；100%表示累计任务构成，不是整步墙钟份额。完整名称、CPU自身调用墙钟及所有阶段见JSON/CSV。',fontsize=9.5)
            fig.text(.035,.026,'阶段由精确CPU提交来源归属，不按加速器执行时间硬切。'+('本图合并反向与梯度规约收尾，原始统计仍分列。' if name=='backward' else '其他更新阶段与未知阶段保留在完整表。'),fontsize=9.5)
            stem=prefix.with_name(prefix.name+'.device%d.hotspots_%s' % (device_index,name))
            for suffix in ('.png','.pdf','.svg'):
                path=Path(str(stem)+suffix)
                if path.exists():raise FileExistsError(path)
                path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160,facecolor='white');paths.append(str(path))
            plt.close(fig)
            record={'device':device,'name':name,'title':title,'merged_phases':list(phases),'actual_tasks':actual,'communication_panel':envelope}
            if envelope['table_kind']=='communication_envelopes':record['communication_envelopes']=envelope
            plots.append(record)
    return {'files':paths,'font':font,'plots':plots,'top_n':top_n,'communication_top_n':communication_top_n,
            'scope':'Display-only phase merging; source statistics are unmodified and contain every row'}
