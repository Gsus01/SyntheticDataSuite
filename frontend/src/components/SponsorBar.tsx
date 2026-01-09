'use client';

import Image from 'next/image';

export default function SponsorBar() {
    return (
        <div className="flex w-full shrink-0 items-center justify-center border-t border-gray-200 bg-gray-50 py-3">
            <div className="container mx-auto flex items-center justify-center px-6">
                <div className="relative h-10 w-full max-w-4xl">
                    <Image
                        src="/sponsors/bannerIA3.png"
                        alt="Colaboradores y Patrocinadores"
                        fill
                        className="object-contain"
                        sizes="(max-width: 1200px) 100vw, 1200px"
                        priority
                        unoptimized
                    />
                </div>
            </div>
        </div>
    );
}

