from steamwebapi.api import ISteamUser, IPlayerService, ISteamUserStats
steamuserinfo = ISteamUser(steam_api_key='DF0E97F40E0BEE667B0BF01B6626A9DA')
steamid = steamuserinfo.resolve_vanity_url("profileURL")['response']['steamid']
print(steamuserinfo.resolve_vanity_url("profileURL"))
