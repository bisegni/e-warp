#![cfg_attr(feature = "release_bundle", windows_subsystem = "windows")]

use anyhow::Result;
use warp_core::channel::{
    Channel, ChannelConfig, ChannelState, OzConfig, ProductProfile, WarpServerConfig,
};
use warp_core::AppId;

fn main() -> Result<()> {
    ChannelState::set(
        ChannelState::new(
            Channel::Oss,
            ChannelConfig {
                app_id: AppId::new("offline", "warp", "EWarp"),
                logfile_name: "ewarp.log".into(),
                server_config: WarpServerConfig::standalone(),
                oz_config: OzConfig::standalone(),
                telemetry_config: None,
                crash_reporting_config: None,
                autoupdate_config: None,
                mcp_static_config: None,
            },
        )
        .with_product_profile(ProductProfile::standalone()),
    );

    warp::run()
}

#[cfg(all(not(feature = "extern_plist"), target_os = "macos"))]
embed_plist::embed_info_plist_bytes!(r#"
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0"><dict>
    <key>CFBundleDisplayName</key><string>eWarp</string>
    <key>CFBundleExecutable</key><string>ewarp</string>
    <key>CFBundleIdentifier</key><string>dev.warp.EWarp</string>
    <key>CFBundleName</key><string>eWarp</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
    <key>NSHighResolutionCapable</key><true/>
    </dict></plist>
"#.as_bytes());
