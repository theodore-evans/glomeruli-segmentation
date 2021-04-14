import json

json_file = open('./outputs/glomeruli_polygons.json')

data = json.load(json_file)

annotation_type = "glomeruli"

xml_annots = open('./annots.xml', 'w')

xml_annots.write('<?xml version="1.0"?>\n')
xml_annots.write('<ASAP_Annotations>\n')
xml_annots.write('	<Annotations>\n')

for item in data['items']:
    xml_annots.write(f'		<Annotation Name="{item["name"]}" Type="Polygon" PartOfGroup="{annotation_type}" Color="#F4FA58">\n')
    xml_annots.write('			<Coordinates>\n')
    for i, coor in enumerate(item['coordinates']):
        xml_annots.write(f'				<Coordinate Order="{i}" X="{coor[0]}" Y="{coor[1]}" />\n')
    xml_annots.write('			</Coordinates>\n')
    xml_annots.write('		</Annotation>\n')

xml_annots.write('	</Annotations>\n')
xml_annots.write('	<AnnotationGroups>\n')
xml_annots.write('		<Group Name="metastases" PartOfGroup="None" Color="#ff0000">\n')
xml_annots.write('			<Attributes />\n')
xml_annots.write('		</Group>\n')
xml_annots.write('	</AnnotationGroups>\n')
xml_annots.write('</ASAP_Annotations>\n')

xml_annots.close()