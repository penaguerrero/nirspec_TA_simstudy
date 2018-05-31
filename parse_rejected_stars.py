import os
from pprint import pprint



# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Feb 2016 - Version 1.0: initial version completed


def mkcentroids(s, delim=' '):
    output = {}
    data = s.split(delim)
    for part in data:
        k, v = part.partition('=')[0::2]
        k = k.rstrip()
        try:
            v = float(v.split()[0])
        except AttributeError as e:
            print(e)
        output[k] = float(v)

    return output


def load_rejected_stars(filename):
    triggers = dict(
        section_header='- Rejected stars -',
        section_start='Centroid',
        inner_lsi='(least_squares_iterate):',
        inner_iter='iteration',
        outer_comment='#',
        outer_marker='*'
    )
    section_header = '- Rejected stars -'
    section_start = 'Centroid'
    section_count = 0
    read_in = False
    centroid_windows = {}

    with open(filename, 'r') as datafile:
        centroid_key = ''
        for line in datafile:
            line = line.strip().lstrip('# ').rstrip()

            if not line or line.startswith('*'):
                continue

            if section_count > 1:
                break

            if section_header in line:
                section_count += 1
                continue

            if section_start in line:
                read_in = True
                centroid_key = line[len(line) - 2]
                centroid_windows[centroid_key] = {}
                continue

            if read_in:
                line = line.replace(triggers['inner_lsi'], '')
                line = line.strip()
                if triggers['inner_iter'] in line:
                    centroid_windows[centroid_key]['iteration'] = int(line[len(line) - 1])
                    # Forget it...
                    #centroid_windows[centroid_key]['units'] = 'arcsecs'
                else:
                    for key, value in mkcentroids(line, '   ').items():
                        centroid_windows[centroid_key][key] = value

    return centroid_windows


def show_rejected_stars(d):
    assert isinstance(d, dict)
    for centroid, subdict in sorted(d.items()):
        print('# Centroid {0}'.format(centroid))
        for key, value in sorted(subdict.items()):
            print('{1:<20s}= {2:<20}'.format(centroid, key, value))
        print('')

if __name__ == '__main__':
    out = load_rejected_stars('./input_file.txt')
    show_rejected_stars(out)
