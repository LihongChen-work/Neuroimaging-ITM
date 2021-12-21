from pyecharts import options as opts
from pyecharts.charts import Graph, Page

def gg(list):
    categories = [{}, {'name': 'DAT'}, {'name': 'IND'}, {'name': 'MOD'}, {'name': 'MOM'}, {'name': 'ROT'},
                  {'name': 'TSK'}]
    nodes = []
    for senlist in list:
        for sen in senlist:
            slist = sen.split("++")
            if(slist[1])=='DAT':
                nodes.append({"name": slist[0], "symbolSize": 10, "category": 1})
            if (slist[1]) == 'IND':
                nodes.append({"name": slist[0], "symbolSize": 10, "category": 2})
            if (slist[1]) == 'MOD':
                nodes.append({"name": slist[0], "symbolSize": 10, "category": 3})
            if (slist[1]) == 'MOM':
                nodes.append({"name": slist[0], "symbolSize": 10, "category": 4})
            if (slist[1]) == 'ROT':
                nodes.append({"name": slist[0], "symbolSize": 10, "category": 5})
            if (slist[1]) == 'TSK':
                nodes.append({"name": slist[0], "symbolSize": 10, "category": 6})






   # {"name": "结点1", "symbolSize": 10, "category": 3},  # categories[3]='类目3'

    links = []
    for i in nodes:
        for j in nodes:
            links.append({"source": i.get("name"), "target": j.get("name"), "value": "weight"})
    c = (
        Graph(init_opts=opts.InitOpts(width="1000px", height="600px"))
            .add("", nodes, links, categories=categories, repulsion=8000,
                 # layout="circular",
                 is_rotate_label=True,
                 linestyle_opts=opts.LineStyleOpts(curve=0.3),
                 label_opts=opts.LabelOpts(position="right"))
            .set_global_opts(title_opts=opts.TitleOpts(title="Graph-基本示例"), )
    )
    c.render()




