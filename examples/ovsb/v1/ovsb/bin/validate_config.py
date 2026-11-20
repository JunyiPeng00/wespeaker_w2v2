"""Validate a recipe config and print it.

Usage::

    python -m ovsb.bin.validate_config conf/stage2_v9c.yaml
    python -m ovsb.bin.validate_config conf/cont_c40.yaml \\
        --set run.exp_dir=exp/trial --view core
    python -m ovsb.bin.validate_config <exp_dir>/config.yaml \\
        --legacy conf/<the conf of that run>.yaml

Exit status 0 = valid, 2 = invalid (the message names the dotted path).
"""

import argparse
import sys
from typing import List, Optional

import yaml

from ovsb.config import core_view, legacy, load
from ovsb.config.schema import ConfigError


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('config', help='recipe config (YAML)')
    parser.add_argument('--legacy', metavar='AUTHORED_CONF', default=None,
                        help='the file is a config saved by the original '
                        'wespeaker-schema trainer; AUTHORED_CONF is the '
                        'hand-written conf of that run, which gives the '
                        'arm order (a saved config has sorted keys)')
    parser.add_argument('--set', dest='overrides', action='append',
                        default=[], metavar='PATH=VALUE',
                        help='override, e.g. run.seed=1 (repeatable)')
    parser.add_argument('--resolve', action='store_true',
                        help='fill the derived values (reads the training '
                        'list unless --num_class is given)')
    parser.add_argument('--num_class', type=int, default=None,
                        help='class count for --resolve')
    parser.add_argument('--view', choices=('recipe', 'core', 'none'),
                        default='recipe',
                        help='what to print: the recipe config, the '
                        'wespeaker-schema view, or nothing')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = get_args(argv)
    try:
        if args.legacy:
            if args.overrides:
                raise ConfigError('--set is not available with --legacy')
            cfg = legacy.load_legacy(args.config, args.legacy)
        else:
            cfg = load.load(args.config, args.overrides)
        if args.resolve or args.num_class is not None:
            cfg = load.resolve(cfg, num_class=args.num_class)
    except ConfigError as exc:
        print('INVALID %s\n%s' % (args.config, exc), file=sys.stderr)
        return 2
    if args.view == 'recipe':
        sys.stdout.write(load.dump(cfg))
    elif args.view == 'core':
        sys.stdout.write(yaml.safe_dump(core_view.to_core_dict(cfg),
                                        sort_keys=False))
    print('OK %s' % args.config, file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
