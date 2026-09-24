"""Competition boundaries shared by ingestion and statistical code.

Legacy untagged tables remain readable only by the old Premier scripts.
The two-league entry point always requires explicit competition and team IDs.
"""
LEAGUES = {'ENG_PL': 'E0', 'COL_A': None}

def validate_competition(frame, expected=None, required=False):
    if expected is not None and expected not in LEAGUES:
        raise ValueError('Competicion desconocida')
    if 'competition_id' not in frame:
        if required or expected is not None:
            raise ValueError('competition_id obligatorio; no se infiere la liga')
        if 'Div' in frame and set(frame.Div.dropna()) != {'E0'}:
            raise ValueError('Datos legacy solo Premier; usar entrada por liga')
        for c in ['home', 'away']:
            if c in frame and frame[c].astype(str).str.contains('::', regex=False).any():
                raise ValueError('IDs con namespace requieren competition_id')
        return None
    ids = frame.competition_id
    if ids.isna().any() or ids.nunique() != 1:
        raise ValueError('Mezcla de ligas o competicion ausente')
    league = ids.iloc[0]
    if league not in LEAGUES or (expected is not None and league != expected):
        raise ValueError('Competicion incompatible')
    if 'Div' in frame and (frame.Div.isna().any() or not frame.Div.eq(LEAGUES[league]).all()):
        raise ValueError('Div no corresponde a competition_id')
    if 'phase' in frame and (not frame.phase.eq('regular_season').all()):
        raise ValueError('No mezclar fase regular con copas o play-offs')
    for c in ['home', 'away']:
        if c not in frame or frame[c].isna().any() or (not frame[c].astype(str).str.startswith(league + '::').all()):
            raise ValueError('Equipo ajeno a la liga o ID sin namespace')
    if 'match_id' in frame:
        if frame.match_id.isna().any() or not frame.match_id.astype(str).str.startswith(league + '::').all():
            raise ValueError('Partido ajeno a la liga')
    return league

def validate_fit_teams(fit, home, away):
    league = fit.get('competition_id')
    if league:
        if league not in LEAGUES or any((not isinstance(t, str) or not t.startswith(league + '::') for t in [home, away])):
            raise ValueError('No se puede predecir otra liga con estos ratings')
    elif any(('::' in str(t) for t in [home, away])):
        raise ValueError('Ratings legacy no admiten equipos con namespace')
