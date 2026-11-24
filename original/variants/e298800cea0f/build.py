import main,results,end
ctx=vars(main)
results.results(ctx)
end.end(ctx)
main.doc.save(main.O/'main.docx')
print(main.O/'main.docx',len(main.records),'data rows')
